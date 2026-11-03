import { AsyncLocalStorage } from "node:async_hooks";
import z from "@deepseek-ai/schemastery";
import { credentialKey, credentialRef } from "@deepseek-ai/dsh-credentials";
import { launchEnvironmentOf } from "@deepseek-ai/dsh-launch-environment";

//#region constants
/** Plugin identity for the loader and its owned settings namespace. */
const name = "llm-github-copilot";

/**
 * No service inject. This plugin no longer registers an LLM adapter: it
 * provisions the credential the harness's own Copilot route consumes. The
 * services it touches (`credentials`, `commands`, `connection`, `settings`)
 * are resolved dynamically so a composition lacking them still loads.
 */
const inject = [];
const NS = "llm-github-copilot";

const DISPLAY_NAME = "GitHub Copilot";

/**
 * The credential record this plugin writes, and the harness route that reads
 * it. The scope is `llm-pi-ai`'s own `RECORD_SCOPE` and the id is its pi-ai
 * provider id, because the record's format belongs to that plugin — see
 * `docs/adr/0002-narrow-to-credential-provider.md`.
 */
const PI_AI_RECORD_SCOPE = "llm-pi-ai";
const PI_AI_PROVIDER = "github-copilot";

/**
 * Credential reference holding the long-lived GitHub OAuth token. The grant
 * record is the store of record; this reference remains supported as an
 * ambient input (`export GITHUB_COPILOT_OAUTH_TOKEN=…`) and is seeded into the
 * grant on activation.
 */
const DEFAULT_OAUTH_TOKEN_ENV = "GITHUB_COPILOT_OAUTH_TOKEN";

/** GitHub OAuth device-flow endpoints (github.com). */
const DEVICE_CODE_URL = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token";
const VERIFICATION_URI = "https://github.com/login/device";

/** VS Code's public GitHub App client id — produces ghu_* tokens that can be exchanged. */
const OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98";
const OAUTH_SCOPE = "read:user";

/** Identity headers matching OpenCode's GitHub Copilot integration. */
const EDITOR_VERSION = "vscode/1.107.0";
const EDITOR_PLUGIN_VERSION = "copilot-chat/0.35.0";
const EXCHANGE_USER_AGENT = "GitHubCopilotChat/0.35.0";
const COPILOT_INTEGRATION_ID = "vscode-chat";
const OPENAI_INTENT = "conversation-edits";
const COPILOT_HEADERS = Object.freeze({
  "User-Agent": EXCHANGE_USER_AGENT,
  "Editor-Version": EDITOR_VERSION,
  "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
  "Copilot-Integration-Id": COPILOT_INTEGRATION_ID
});
//#endregion

//#region settings schema
/**
 * The Loader-owned live configuration. It configures only how the credential
 * is addressed: everything about models, endpoints, and requests belongs to
 * the harness route that consumes the credential.
 */
const Config = z.object({
  /**
   * Credential reference for the long-lived GitHub OAuth token. Reading it
   * supports an ambient `export GITHUB_COPILOT_OAUTH_TOKEN=…`; the device flow
   * writes through it as well, so an existing deployment keeps working.
   */
  oauthTokenEnv: z.string().role("credential-ref").default(DEFAULT_OAUTH_TOKEN_ENV).volatile()
});
//#endregion

//#region transport
/**
 * GitHub request transport for the device flow, honoring the ambient proxy.
 *
 * The undici `ProxyAgent` routes each request through
 * `https_proxy`/`HTTPS_PROXY`/`http_proxy`/`HTTP_PROXY` (+ `NO_PROXY`) at
 * runtime — no launch flag needed — and falls back to the global fetch when no
 * proxy applies. A hand-rolled CONNECT tunnel is deliberately NOT used: on
 * mixed-mode proxies it bypasses the proxy egress, which GitHub treats as a
 * different client.
 */
import undici from "undici";
const ProxyAgent = undici.ProxyAgent;
let proxyDispatcherCache;
function resolveHttpProxy() {
  const order = ["https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"];
  for (const key of order) {
    const value = process.env[key];
    if (!value) continue;
    try {
      const parsed = new URL(value);
      if (parsed.protocol === "http:" || parsed.protocol === "https:") return value;
    } catch {}
  }
  return void 0;
}
function isProxyBypassed(hostname) {
  const raw = process.env.no_proxy ?? process.env.NO_PROXY;
  if (!raw) return false;
  const host = hostname.toLowerCase();
  return raw.split(",").map((part) => part.trim().toLowerCase()).filter(Boolean).some((pattern) => {
    if (pattern === "*") return true;
    if (pattern.includes("/")) return false; // CIDR ranges are not supported
    const base = pattern.startsWith(".") ? pattern.slice(1) : pattern;
    return host === base || host.endsWith("." + base);
  });
}
/** A cached undici ProxyAgent for the current proxy URL, or undefined when no proxy applies. */
function proxyDispatcher() {
  const url = resolveHttpProxy();
  if (url === void 0) return void 0;
  if (proxyDispatcherCache !== void 0 && proxyDispatcherCache.url === url) return proxyDispatcherCache.agent;
  const agent = new ProxyAgent(url);
  proxyDispatcherCache = { url, agent };
  return agent;
}
async function copilotFetch(url, init = {}) {
  const target = new URL(url);
  const dispatcher = target.protocol === "https:" && !isProxyBypassed(target.hostname) ? proxyDispatcher() : void 0;
  if (dispatcher === void 0) return fetch(url, init);
  return fetch(url, { ...init, dispatcher });
}
//#endregion

//#region wire alignment
/**
 * Align outgoing GitHub Copilot HTTP requests with OpenCode's wire identity
 * and X-Initiator quota attribution.
 *
 * DeepSeek Harness's `dsh-llm-pi-ai` appends `attributionHeaders()` (`user-agent:
 * deepseek-harness/...`) after `model.headers`, which overwrites `pi-ai`'s
 * `User-Agent: GitHubCopilotChat/0.35.0` on every inference and model-listing
 * request. It also strips `GenerateOptions.purpose` (`session-title`,
 * `compaction`) and `Message.source.kind` (`runtime-context`, `time-context`),
 * causing `pi-ai`'s `inferCopilotInitiator()` to misclassify internal agent
 * steps and background tasks as `X-Initiator: user` and burn Premium Request
 * quota.
 *
 * Two cooperating layers fix both without forking Harness or `pi-ai`:
 * 1. A Cordis `llm/stream` waterfall listener records the true initiator and
 *    image presence from DSH's typed `GenerateOptions` in `AsyncLocalStorage`.
 * 2. A `globalThis.fetch` wrapper shapes requests targeting Copilot endpoints
 *    (or carrying Copilot integration headers) so their headers match OpenCode.
 */
const copilotCallStore = new AsyncLocalStorage();

const RESPONSES_AGENT_ITEM_TYPES = new Set([
  "file_search_call",
  "computer_call",
  "computer_call_output",
  "web_search_call",
  "function_call",
  "function_call_output",
  "image_generation_call",
  "code_interpreter_call",
  "local_shell_call",
  "local_shell_call_output",
  "mcp_list_tools",
  "mcp_approval_request",
  "mcp_approval_response",
  "mcp_call",
  "reasoning",
  "custom_tool_call",
  "custom_tool_call_output"
]);

const STRIPPED_HEADER_NAMES = new Set([
  "anthropic-dangerous-direct-browser-access",
  "session_id",
  "x-session-affinity",
  "x-client-request-id"
]);

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Recognize loop-injected DSH context snapshots that `dsh-llm-pi-ai` projects
 * as `role: "user"` messages after tool results.
 */
function isSyntheticDshContextText(text) {
  if (typeof text !== "string") return false;
  const trimmed = text.trimStart();
  return (
    trimmed.startsWith("Current runtime context:") ||
    trimmed.startsWith("Time sampled while preparing turn ")
  );
}

function isSyntheticDshUserContent(content) {
  if (typeof content === "string") return isSyntheticDshContextText(content);
  if (!Array.isArray(content) || content.length === 0) return false;
  return content.every(
    (part) =>
      isRecord(part) &&
      (part.type === "text" || part.type === "input_text") &&
      isSyntheticDshContextText(part.text)
  );
}

/**
 * Whether a DSH `RequestMessage` was authored by the human user rather than
 * injected by a Harness loop or context plugin (`runtime-context`,
 * `time-context`, `dsh-session-title-llm`, `compaction`, tool contexts, etc.).
 */
function isHumanUserMessage(message) {
  if (!isRecord(message) || message.role !== "user") return false;
  const kind = isRecord(message.source) ? message.source.kind : void 0;
  if (kind !== void 0 && kind !== "user") return false;
  if (isSyntheticDshUserContent(message.content)) return false;
  return true;
}

/**
 * Determine whether a DSH `llm/stream` request is `"user"` or `"agent"`
 * initiated from full Harness metadata before `dsh-llm-pi-ai` strips it.
 */
function inferDshInitiator(options) {
  if (!isRecord(options)) return "user";
  if (typeof options.purpose === "string" && options.purpose.length > 0) {
    return "agent";
  }
  const messages = Array.isArray(options.messages) ? options.messages : [];
  if (messages.length === 0) return "user";
  let lastAgentOrToolIndex = -1;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const role = messages[i]?.role;
    if (role === "assistant" || role === "tool") {
      lastAgentOrToolIndex = i;
      break;
    }
  }
  if (lastAgentOrToolIndex >= 0) {
    const trailing = messages.slice(lastAgentOrToolIndex + 1);
    return trailing.some(isHumanUserMessage) ? "user" : "agent";
  }
  return messages.some(isHumanUserMessage) ? "user" : "agent";
}

/**
 * Whether DSH `GenerateOptions.messages` carries any non-offloaded image block.
 */
function dshMessagesHaveImage(messages) {
  if (!Array.isArray(messages)) return false;
  return messages.some(
    (msg) =>
      isRecord(msg) &&
      Array.isArray(msg.content) &&
      msg.content.some((block) => isRecord(block) && block.type === "image" && block.offloaded !== true)
  );
}

function lastNonSyntheticEntry(entries) {
  for (let i = entries.length - 1; i >= 0; i -= 1) {
    const entry = entries[i];
    if (isRecord(entry) && entry.role === "user" && isSyntheticDshUserContent(entry.content)) {
      continue;
    }
    return entry;
  }
  return entries.at(-1);
}

/**
 * Inspect the serialized wire body of a Copilot inference call (`/responses`,
 * `/chat/completions`, `/v1/messages`), mirroring OpenCode's body metadata
 * detector while also accounting for DSH's synthetic context messages.
 */
function inspectCopilotWireBody(url, rawBody) {
  let body = rawBody;
  if (typeof rawBody === "string") {
    try {
      body = JSON.parse(rawBody);
    } catch {
      return { isInference: false, agent: false, vision: false };
    }
  } else if (rawBody instanceof Uint8Array || rawBody instanceof ArrayBuffer) {
    try {
      body = JSON.parse(new TextDecoder().decode(rawBody));
    } catch {
      return { isInference: false, agent: false, vision: false };
    }
  }
  if (!isRecord(body)) return { isInference: false, agent: false, vision: false };

  if (Array.isArray(body.input)) {
    const last = lastNonSyntheticEntry(body.input);
    const isAgentItem = (item) =>
      !isRecord(item) ||
      item.role === "assistant" ||
      item.role === "developer" ||
      item.role === "system" ||
      (typeof item.type === "string" && RESPONSES_AGENT_ITEM_TYPES.has(item.type)) ||
      (item.role !== "user" && item.type !== "message");
    const vision = body.input.some(
      (item) =>
        isRecord(item) &&
        (item.type === "input_image" ||
          (Array.isArray(item.content) &&
            item.content.some(
              (part) => isRecord(part) && (part.type === "input_image" || part.type === "image_url")
            )))
    );
    return {
      isInference: true,
      agent: body.input.length > 0 && isAgentItem(last),
      vision
    };
  }

  if (Array.isArray(body.messages)) {
    const last = lastNonSyntheticEntry(body.messages);
    const isAnthropic = typeof url === "string" && /\/v1\/messages(?:\b|$)/.test(url);
    if (isAnthropic) {
      const content = isRecord(last) && Array.isArray(last.content) ? last.content : void 0;
      const hasHumanPart =
        content === void 0
          ? isRecord(last) && typeof last.content === "string" && !isSyntheticDshContextText(last.content)
          : content.some(
              (part) =>
                isRecord(part) &&
                part.type !== "tool_result" &&
                !(part.type === "text" && isSyntheticDshContextText(part.text))
            );
      const vision = body.messages.some(
        (msg) =>
          isRecord(msg) &&
          Array.isArray(msg.content) &&
          msg.content.some(
            (part) =>
              isRecord(part) &&
              (part.type === "image" ||
                (part.type === "tool_result" &&
                  Array.isArray(part.content) &&
                  part.content.some((nested) => isRecord(nested) && nested.type === "image")))
          )
      );
      return {
        isInference: true,
        agent: body.messages.length > 0 && (!isRecord(last) || last.role !== "user" || !hasHumanPart),
        vision
      };
    }

    const vision = body.messages.some(
      (msg) =>
        isRecord(msg) &&
        Array.isArray(msg.content) &&
        msg.content.some((part) => isRecord(part) && (part.type === "image_url" || part.type === "image"))
    );
    return {
      isInference: true,
      agent: body.messages.length > 0 && (!isRecord(last) || last.role !== "user"),
      vision
    };
  }

  return { isInference: false, agent: false, vision: false };
}

/**
 * Whether a URL points to a GitHub Copilot API or token-exchange endpoint.
 */
function isCopilotUrl(url) {
  try {
    const parsed = url instanceof URL ? url : new URL(String(url));
    const host = parsed.hostname.toLowerCase();
    if (host === "githubcopilot.com" || host.endsWith(".githubcopilot.com") || host.startsWith("copilot-api.")) {
      return true;
    }
    if ((host === "api.github.com" || host.startsWith("api.")) && parsed.pathname.startsWith("/copilot_internal/")) {
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

function hasCopilotMarkerHeaders(headers) {
  const integration = headers.get("copilot-integration-id");
  if (integration === COPILOT_INTEGRATION_ID) return true;
  const plugin = headers.get("editor-plugin-version");
  return typeof plugin === "string" && plugin.startsWith("copilot-chat/");
}

function isInferenceEndpointPath(url) {
  try {
    const pathname = (url instanceof URL ? url : new URL(String(url))).pathname;
    return (
      pathname.endsWith("/chat/completions") ||
      pathname.endsWith("/responses") ||
      pathname.endsWith("/messages")
    );
  } catch {
    return false;
  }
}

/**
 * Align a single outgoing `fetch(input, init)` call with OpenCode's Copilot
 * wire contract when it targets GitHub Copilot. Returns `undefined` when the
 * request does not belong to GitHub Copilot.
 */
function alignCopilotRequest(input, init = {}, store = copilotCallStore.getStore()) {
  const url = input instanceof URL ? input.href : typeof input === "string" ? input : input?.url ?? "";
  const merged = new Headers(input instanceof Request ? input.headers : void 0);
  if (init?.headers !== void 0) {
    for (const [key, value] of new Headers(init.headers).entries()) {
      merged.set(key, value);
    }
  }

  const bodyInfo = inspectCopilotWireBody(url, init?.body);
  const isInference =
    !url.includes("/copilot_internal/") &&
    (bodyInfo.isInference || isInferenceEndpointPath(url));
  const matched =
    isCopilotUrl(url) ||
    hasCopilotMarkerHeaders(merged) ||
    (store !== void 0 && isInference);
  if (!matched) return void 0;

  const existingInitiator = merged.get("x-initiator")?.toLowerCase();
  const existingVision = merged.get("copilot-vision-request");
  const apiKeyHeader = merged.get("x-api-key");
  if (apiKeyHeader && !merged.has("authorization")) {
    merged.set("Authorization", `Bearer ${apiKeyHeader}`);
  }
  merged.delete("x-api-key");

  for (const key of [...merged.keys()]) {
    const lower = key.toLowerCase();
    if (lower.startsWith("x-stainless-") || STRIPPED_HEADER_NAMES.has(lower) || lower === "user-agent") {
      merged.delete(key);
    }
  }

  for (const [key, value] of Object.entries(COPILOT_HEADERS)) {
    merged.set(key, value);
  }

  if (isInference) {
    merged.set("Openai-Intent", OPENAI_INTENT);
    // Escalate to "agent" whenever DSH context, existing headers, or the wire
    // body identifies an agent/tool/background turn; never downgrade to "user".
    const isAgent =
      store?.initiator === "agent" ||
      existingInitiator === "agent" ||
      bodyInfo.agent;
    merged.set("X-Initiator", isAgent ? "agent" : "user");
    const isVision = Boolean(store?.hasImages || bodyInfo.vision || existingVision === "true");
    if (isVision) merged.set("Copilot-Vision-Request", "true");
    else merged.delete("copilot-vision-request");
  }

  let dispatcher = init?.dispatcher;
  if (dispatcher === void 0) {
    try {
      const parsed = new URL(url);
      if (parsed.protocol === "https:" && !isProxyBypassed(parsed.hostname)) {
        dispatcher = proxyDispatcher();
      }
    } catch {}
  }

  return {
    input: input instanceof Request ? new Request(input, { headers: merged }) : input,
    init: {
      ...init,
      headers: merged,
      ...dispatcher === void 0 ? {} : { dispatcher }
    }
  };
}

/**
 * Install the `llm/stream` waterfall tracker and `globalThis.fetch` Copilot
 * shaper for the lifetime of `ctx`.
 */
function installCopilotWireAlignment(ctx) {
  if (typeof ctx.on === "function") {
    ctx.on("llm/stream", (options, next) => {
      if (options?.provider !== PI_AI_PROVIDER) return next();
      const callContext = {
        initiator: inferDshInitiator(options),
        hasImages: dshMessagesHaveImage(options.messages),
        ...options.purpose === void 0 ? {} : { purpose: options.purpose },
        ...options.sessionId === void 0 ? {} : { sessionId: String(options.sessionId) }
      };
      const upstream = copilotCallStore.run(callContext, () => next());
      return {
        [Symbol.asyncIterator]() {
          const iterator = copilotCallStore.run(callContext, () => upstream[Symbol.asyncIterator]());
          return {
            next: () => copilotCallStore.run(callContext, () => iterator.next()),
            return: (value) =>
              iterator.return === void 0
                ? Promise.resolve({ done: true, value })
                : copilotCallStore.run(callContext, () => iterator.return(value)),
            throw: (error) =>
              iterator.throw === void 0
                ? Promise.reject(error)
                : copilotCallStore.run(callContext, () => iterator.throw(error))
          };
        }
      };
    });
  }

  ctx.effect(() => {
    const originalFetch = globalThis.fetch;
    if (typeof originalFetch !== "function") return () => {};
    const alignedFetch = async (input, init) => {
      let bodyForInspection = init?.body;
      if (bodyForInspection === void 0 && input instanceof Request && input.method === "POST" && input.body !== null) {
        try {
          bodyForInspection = await input.clone().text();
        } catch {}
      }
      const aligned = alignCopilotRequest(
        input,
        bodyForInspection === init?.body ? init : { ...init, body: bodyForInspection }
      );
      if (aligned === void 0) return originalFetch(input, init);
      if (input instanceof Request && init?.body === void 0) {
        const { body: _inspectedBody, ...restInit } = aligned.init;
        return originalFetch(aligned.input, restInit);
      }
      return originalFetch(aligned.input, aligned.init);
    };
    globalThis.fetch = alignedFetch;
    return () => {
      if (globalThis.fetch === alignedFetch) {
        globalThis.fetch = originalFetch;
      }
    };
  }, `${name}: OpenCode wire alignment`);
}
//#endregion
//#region config resolution
/**
 * Resolve the settings section into the one fact this plugin acts on.
 * @param {object} config - the raw settings section.
 * @returns {{ oauthTokenEnv: string }} the resolved credential reference.
 */
function resolveAdapterOptions(config) {
  return {
    oauthTokenEnv: credentialRef(config.oauthTokenEnv ?? DEFAULT_OAUTH_TOKEN_ENV)
  };
}

/**
 * The credential record the harness's Copilot route reads.
 * @returns {import('@deepseek-ai/dsh-credentials').CredentialKey}
 */
function piAiRecordKey() {
  return credentialKey(PI_AI_RECORD_SCOPE, PI_AI_PROVIDER);
}

/**
 * The grant payload a fresh sign-in produces.
 *
 * Only `refresh` carries information. The consuming route performs the Copilot
 * token exchange itself and writes `access`, `expires`, `availableModelIds`,
 * and the account's endpoint back into this same record, so seeding them with
 * empty values is what tells it the grant is unexchanged.
 *
 * @param {string} token - the long-lived GitHub OAuth token.
 * @returns {{ kind: 'grant', payload: object }}
 */
function piAiGrantRecord(token) {
  return { kind: "grant", payload: { type: "oauth", refresh: token, access: "", expires: 0 } };
}

/**
 * The long-lived token inside a stored grant, when the record holds one.
 * @param {object | undefined} record - the stored credential record.
 * @returns {string | undefined} the token, or undefined for any other shape.
 */
function grantToken(record) {
  if (record?.kind !== "grant") return void 0;
  const refresh = record.payload?.refresh;
  return typeof refresh === "string" && refresh.length > 0 ? refresh : void 0;
}

/**
 * The models the consuming route recorded as available to this account. It
 * writes them during its first exchange, so this reads back what the account
 * can actually use without this plugin interrogating any endpoint.
 * @param {object | undefined} record - the stored credential record.
 * @returns {readonly string[] | undefined} model ids, when recorded.
 */
function grantModelIds(record) {
  if (record?.kind !== "grant") return void 0;
  const ids = record.payload?.availableModelIds;
  return Array.isArray(ids) && ids.every((id) => typeof id === "string") ? ids : void 0;
}
//#endregion

//#region device flow
/**
 * A device-flow failure. Carries a code so callers can distinguish transport
 * from server refusal, replacing the `LlmError` this plugin used while it was
 * an LLM adapter — nothing here is an LLM concern any more.
 */
class CopilotAuthError extends Error {
  /**
   * @param {string} message - human-readable failure.
   * @param {string} code - stable machine-readable tag.
   * @param {{ cause?: unknown, status?: number }} [details]
   */
  constructor(message, code, details = {}) {
    super(message, details.cause === void 0 ? void 0 : { cause: details.cause });
    this.name = "CopilotAuthError";
    this.code = code;
    if (details.status !== void 0) this.status = details.status;
  }
}

async function startDeviceFlow() {
  let response;
  try {
    response = await copilotFetch(DEVICE_CODE_URL, {
      method: "POST",
      headers: {
        accept: "application/json",
        "content-type": "application/x-www-form-urlencoded",
        ...COPILOT_HEADERS
      },
      body: new URLSearchParams({ client_id: OAUTH_CLIENT_ID, scope: OAUTH_SCOPE }).toString()
    });
  } catch (error) {
    throw new CopilotAuthError("failed to start GitHub device authorization", "TRANSPORT", { cause: error });
  }
  if (!response.ok) throw new CopilotAuthError(`GitHub device authorization failed (HTTP ${response.status})`, response.status >= 500 ? "SERVER" : "TRANSPORT", { status: response.status });
  const data = await response.json();
  const deviceCode = data.device_code;
  const userCode = data.user_code;
  if (typeof deviceCode !== "string" || deviceCode.length === 0 || typeof userCode !== "string" || userCode.length === 0) throw new CopilotAuthError("GitHub did not return a device code", "TRANSPORT");
  return {
    deviceCode,
    userCode,
    verificationUri: typeof data.verification_uri === "string" && data.verification_uri.length > 0 ? data.verification_uri : VERIFICATION_URI,
    interval: Math.max(Number(data.interval) || 5, 1),
    expiresIn: Number(data.expires_in) || 900
  };
}
async function pollDeviceFlow(device) {
  const body = new URLSearchParams({
    client_id: OAUTH_CLIENT_ID,
    device_code: device.deviceCode,
    grant_type: "urn:ietf:params:oauth:grant-type:device_code"
  }).toString();
  let response;
  try {
    response = await copilotFetch(ACCESS_TOKEN_URL, {
      method: "POST",
      headers: {
        accept: "application/json",
        "content-type": "application/x-www-form-urlencoded",
        ...COPILOT_HEADERS
      },
      body
    });
  } catch {
    return { state: "retry" };
  }
  if (!response.ok) return { state: "retry" };
  const data = await response.json();
  if (typeof data.access_token === "string" && data.access_token.length > 0) return { state: "done", token: data.access_token };
  switch (data.error) {
    case "authorization_pending": return { state: "retry" };
    case "slow_down": return { state: "slow-down" };
    case "expired_token": return { state: "expired" };
    case "access_denied": return { state: "denied" };
    default: return { state: "error", message: data.error ?? "unknown" };
  }
}
//#endregion

//#region apply
/**
 * Provision the credential the harness's own Copilot route consumes.
 *
 * This plugin registers no LLM adapter. It owns the GitHub device flow and
 * writes its result as a grant record that `llm-pi-ai` reads for its
 * `github-copilot` provider; that route then performs the Copilot token
 * exchange, refreshes it, derives the account's endpoint, and serves every
 * request. See `docs/adr/0002-narrow-to-credential-provider.md`.
 */
function apply(ctx, config) {
  let lastRaw;
  let lastGood;
  const options = () => {
    const raw = config.oauthTokenEnv.get();
    if (raw === lastRaw && lastGood !== void 0) return lastGood;
    try {
      const next = resolveAdapterOptions({ oauthTokenEnv: raw });
      lastRaw = raw;
      lastGood = next;
      return next;
    } catch (error) {
      if (lastGood === void 0) throw error;
      lastRaw = raw;
      ctx.logger.error(`${name}: keeping the last good configuration after an invalid credential reference`);
      ctx.logger.error(error);
      return lastGood;
    }
  };
  options();
  installCopilotWireAlignment(ctx);

  // ── credential plane ─────────────────────────────────────────────────────
  /**
   * The grant record, when the credentials service is mounted. A composition
   * without one genuinely holds no credential, so reads answer "nothing".
   */
  const readGrant = async () => {
    const credentials = ctx.get("credentials");
    if (credentials === void 0) return void 0;
    return credentials.readRecord(piAiRecordKey());
  };

  /**
   * The long-lived token, from the grant first and the configured reference
   * second. The reference keeps an ambient `export GITHUB_COPILOT_OAUTH_TOKEN=…`
   * working, and is what an installation predating the grant still holds.
   */
  const resolveRawOAuthToken = async () => {
    const fromGrant = grantToken(await readGrant());
    if (fromGrant !== void 0) return fromGrant;
    const ref = options().oauthTokenEnv;
    const credentials = ctx.get("credentials");
    if (credentials !== void 0) {
      const hit = await credentials.resolve(ref);
      if (hit !== void 0 && hit.value.length > 0) return hit.value;
    }
    const ambient = launchEnvironmentOf(ctx).get(ref);
    return ambient !== void 0 && ambient.value.length > 0 ? ambient.value : void 0;
  };

  /**
   * Publish a token to the consuming route, and to this plugin's own reference
   * so an ambient deployment and a signed-in one look alike.
   *
   * The write is read back and the token compared, because the record's format
   * belongs to another plugin: a shape it stops accepting must fail loudly here
   * rather than leave a credential that silently authenticates nothing.
   *
   * The reference write is skipped when the launching environment supplies the
   * reference read-only: Harness releases before 0.2.1 reject that `set`, and
   * the environment already resolves the reference, so the grant alone suffices.
   */
  const storeRawOAuthToken = async (token) => {
    const credentials = ctx.get("credentials");
    if (credentials === void 0) {
      throw new Error(`${name}: signing in needs the credentials service; mount dsh-credentials-local`);
    }
    await credentials.modifyRecord(piAiRecordKey(), async () => piAiGrantRecord(token));
    if (grantToken(await credentials.readRecord(piAiRecordKey())) !== token) {
      throw new Error(
        `${name}: the credential record at ${PI_AI_RECORD_SCOPE}/${PI_AI_PROVIDER} did not survive the write;`
        + " the consuming plugin may have changed its grant format",
      );
    }
    const ref = options().oauthTokenEnv;
    if ((await credentials.describe(ref)).writable) await credentials.set(ref, token);
  };

  /**
   * Adopt a token this plugin can already see into the grant the route reads.
   * Runs once the credential service is available so an ambient token, or an
   * installation predating the grant, authenticates without a second sign-in.
   *
   * This is scoped to `credentials` rather than run at activation: the plugin
   * declares no service inject, so it activates before the credential plane
   * exists and an unscoped read would see nothing and silently give up.
   */
  const seedGrantFromReference = async () => {
    try {
      if (grantToken(await readGrant()) !== void 0) return;
      const existing = await resolveRawOAuthToken();
      if (existing === void 0) return;
      await storeRawOAuthToken(existing);
      ctx.logger.info(
        `${name}: adopted the existing ${options().oauthTokenEnv} token into the`
        + ` ${PI_AI_RECORD_SCOPE}/${PI_AI_PROVIDER} credential`,
      );
    } catch (error) {
      ctx.logger.warn(`${name}: could not adopt an existing token; sign in again to publish one`);
      ctx.logger.warn(error);
    }
  };
  ctx.inject(["credentials"], (credentialCtx) => {
    credentialCtx.effect(() => {
      void seedGrantFromReference();
      return () => {};
    }, `${name}: adopt an existing token`);
  });

  // dsh-settings owns optional presentation policy. The Loader owns this
  // plugin's volatile Config, so the device-flow controller reads its reference
  // directly and stays usable in compositions without a settings service.
  ctx.inject(["settings"], (settingsCtx) => {
    settingsCtx.effect(
      () => settingsCtx.settings.configure({ auto: false }, ctx.fiber),
      `${name}: custom settings page`,
    );
  });

  // ── shared OAuth controller (commands + Web settings page) ────────────────
  const activeTimers = /* @__PURE__ */ new Set();
  let authGeneration = 0;
  let authFlow;
  const clearAuthTimers = () => {
    for (const timer of activeTimers) clearTimeout(timer);
    activeTimers.clear();
  };
  ctx.effect(() => () => {
    authGeneration += 1;
    clearAuthTimers();
  });
  const publicFlow = (flow) => flow === void 0 ? void 0 : {
    state: flow.state,
    ...flow.verificationUri === void 0 ? {} : { verificationUri: flow.verificationUri },
    ...flow.userCode === void 0 ? {} : { userCode: flow.userCode },
    ...flow.expiresAt === void 0 ? {} : { expiresAt: flow.expiresAt },
    ...flow.message === void 0 ? {} : { message: flow.message }
  };
  const schedulePoll = (device) => {
    clearAuthTimers();
    const generation = ++authGeneration;
    let interval = device.interval;
    const deadline = Date.now() + device.expiresIn * 1000;
    authFlow = {
      state: "pending",
      verificationUri: device.verificationUri,
      userCode: device.userCode,
      expiresAt: deadline
    };
    let timer;
    const finish = (state, message) => {
      if (generation !== authGeneration) return;
      authFlow = { state, ...message === void 0 ? {} : { message } };
    };
    const tick = async () => {
      activeTimers.delete(timer);
      if (generation !== authGeneration) return;
      if (Date.now() > deadline) {
        finish("expired", "The device code expired. Start sign-in again.");
        ctx.logger.warn(`${name}: device authorization expired`);
        return;
      }
      const result = await pollDeviceFlow(device);
      if (generation !== authGeneration) return;
      if (result.state === "done") {
        try {
          await storeRawOAuthToken(result.token);
          finish("authenticated");
          ctx.logger.info(
            `${name}: GitHub Copilot sign-in completed; the ${PI_AI_PROVIDER} route can now authenticate`,
          );
        } catch (error) {
          finish("error", error instanceof Error ? error.message : String(error));
          ctx.logger.error(`${name}: failed to store the Copilot credential`);
          ctx.logger.error(error);
        }
        return;
      }
      switch (result.state) {
        case "slow-down": interval = Math.max(interval + 5, 1); break;
        case "expired": finish("expired", "The device code expired. Start sign-in again."); return;
        case "denied": finish("denied", "GitHub authorization was denied."); return;
        case "error": finish("error", result.message); return;
        default: break;
      }
      timer = setTimeout(tick, interval * 1000);
      activeTimers.add(timer);
    };
    timer = setTimeout(tick, interval * 1000);
    activeTimers.add(timer);
  };
  const authStatus = async () => {
    const record = await readGrant();
    const stored = grantToken(record);
    const raw = stored ?? await resolveRawOAuthToken();
    if (raw === void 0) return {
      authenticated: false,
      credential: options().oauthTokenEnv,
      provider: PI_AI_PROVIDER,
      ...publicFlow(authFlow) ?? { state: "signed-out" }
    };
    // Model ids are read back from the record the consuming route maintains,
    // never interrogated here: this plugin owns no catalog.
    const modelIds = grantModelIds(record);
    return {
      authenticated: true,
      state: "authenticated",
      credential: options().oauthTokenEnv,
      provider: PI_AI_PROVIDER,
      ...stored === void 0 ? { credentialRemaining: true } : {},
      ...modelIds === void 0 ? {} : { modelCount: modelIds.length }
    };
  };
  const beginLogin = async () => {
    const existing = await resolveRawOAuthToken();
    if (existing !== void 0) return authStatus();
    if (authFlow?.state === "pending" && typeof authFlow.expiresAt === "number" && Date.now() < authFlow.expiresAt) {
      return { authenticated: false, ...publicFlow(authFlow) };
    }
    const device = await startDeviceFlow();
    schedulePoll(device);
    return { authenticated: false, ...publicFlow(authFlow) };
  };
  const logout = async () => {
    const ref = options().oauthTokenEnv;
    const credentials = ctx.get("credentials");
    if (credentials === void 0) throw new Error(`${name}: signing out needs the credentials service`);
    authGeneration += 1;
    clearAuthTimers();
    authFlow = { state: "signed-out" };
    // Clear managed storage, never the launching environment. Older providers
    // expose ambient references read-only; newer ones unset a file override but
    // still fall back to the environment. Report the remaining token honestly.
    await credentials.deleteRecord(piAiRecordKey());
    if ((await credentials.describe(ref)).writable && (await credentials.resolve(ref)) !== void 0) {
      await credentials.unset(ref);
    }
    return authStatus();
  };

  // ── Web settings API (optional; only mounted by the web profile) ──────────
  // `ctx.webServer.register()` is a raw, unauthenticated HTTP route: it is not
  // reachable through the browser's shared `/api` channel at all, and nothing
  // applies the harness's Host/Origin fence or browser-session check to it.
  // Login and logout must instead go through `ctx.connection.fetch.register()`,
  // whose `path` is a literal absolute path under `/api` and whose handler runs
  // only once Connection's trust and authentication policy has already passed.
  const jsonResponse = (status, payload) => new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" }
  });
  const fetchAction = (action) => async () => {
    try {
      return jsonResponse(200, { ok: true, value: await action() });
    } catch (error) {
      return jsonResponse(500, { ok: false, error: error instanceof Error ? error.message : String(error) });
    }
  };
  ctx.inject(["connection"], (cctx) => {
    cctx.effect(() => {
      const disposers = [
        cctx.connection.fetch.register({
          path: "/api/github-copilot-auth/status",
          methods: ["GET"],
          requestBody: "buffered",
          fetch: fetchAction(authStatus)
        }),
        cctx.connection.fetch.register({
          path: "/api/github-copilot-auth/login",
          methods: ["POST"],
          requestBody: "buffered",
          fetch: fetchAction(beginLogin)
        }),
        cctx.connection.fetch.register({
          path: "/api/github-copilot-auth/logout",
          methods: ["POST"],
          requestBody: "buffered",
          fetch: fetchAction(logout)
        })
      ];
      return async () => {
        for (const dispose of disposers.reverse()) await dispose();
      };
    }, `${name}: Web OAuth routes`);
  });

  // ── slash commands (fallback for non-Web surfaces) ───────────────────────
  const remainingCredentialNotice = (ref) => `${ref} still supplies a token outside the local grant.`
    + " Remove it from the launching environment, .env, or external credential source and restart DSH to finish signing out.";
  ctx.inject(["commands"], (cctx) => {
    cctx.commands.register({
      name: "copilot-login",
      description: "Sign in to GitHub Copilot via the device flow",
      handler: async () => {
        try {
          const status = await beginLogin();
          if (status.credentialRemaining) return { kind: "success", text: remainingCredentialNotice(status.credential) };
          if (status.authenticated) return {
            kind: "success",
            text: `GitHub Copilot is already authenticated. Run /copilot-status for details.`
          };
          return {
            kind: "success",
            text: [
              "GitHub Copilot sign-in",
              `1. Open this URL in your browser: ${status.verificationUri}`,
              `2. Enter this code: ${status.userCode}`,
              "3. Authorize the GitHub App (VS Code) and complete sign-in.",
              "",
              "Polling continues in the background; your credential is stored automatically once you authorize.",
              `Then select a model under the "${PI_AI_PROVIDER}" provider in Settings → Models.`
            ].join("\n")
          };
        } catch (error) {
          return { kind: "error", text: error instanceof Error ? error.message : String(error) };
        }
      }
    });
    cctx.commands.register({
      name: "copilot-status",
      description: "Show GitHub Copilot authentication status",
      handler: async () => {
        try {
          const status = await authStatus();
          if (status.credentialRemaining) return { kind: "success", text: remainingCredentialNotice(status.credential) };
          if (!status.authenticated) return {
            kind: "success",
            text: `GitHub Copilot is NOT signed in. Run /copilot-login to sign in, or set ${options().oauthTokenEnv}.`
          };
          return {
            kind: "success",
            text: [
              `GitHub Copilot: authenticated.`,
              `Credential published to the "${status.provider}" provider.`,
              status.modelCount === void 0
                ? "Model list is populated by that provider on its first request."
                : `Models available to this account: ${status.modelCount}`,
              `Select a model under that provider in Settings → Models.`
            ].join("\n")
          };
        } catch (error) {
          return { kind: "error", text: error instanceof Error ? error.message : String(error) };
        }
      }
    });
    cctx.commands.register({
      name: "copilot-logout",
      description: "Sign out of GitHub Copilot and remove the stored credential",
      handler: async () => {
        try {
          const existing = await resolveRawOAuthToken();
          const status = await logout();
          return {
            kind: "success",
            text: status.credentialRemaining
              ? `Local GitHub Copilot grant removed, but ${remainingCredentialNotice(status.credential)}`
              : existing === void 0
                ? "GitHub Copilot is not signed in. Nothing to do."
                : `GitHub Copilot signed out; the "${status.provider}" credential has been removed.`
          };
        } catch (error) {
          return { kind: "error", text: error instanceof Error ? error.message : String(error) };
        }
      }
    });
  });
}
//#endregion



export { COPILOT_HEADERS, COPILOT_INTEGRATION_ID, Config, DEFAULT_OAUTH_TOKEN_ENV, EDITOR_PLUGIN_VERSION, EDITOR_VERSION, EXCHANGE_USER_AGENT, OPENAI_INTENT, PI_AI_PROVIDER, PI_AI_RECORD_SCOPE, alignCopilotRequest, apply, dshMessagesHaveImage, grantModelIds, grantToken, inferDshInitiator, inject, inspectCopilotWireBody, installCopilotWireAlignment, isCopilotUrl, name, piAiGrantRecord, piAiRecordKey, resolveAdapterOptions };
