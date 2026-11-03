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
