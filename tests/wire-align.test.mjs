import test from 'node:test'
import assert from 'node:assert/strict'
import {
  COPILOT_HEADERS,
  EXCHANGE_USER_AGENT,
  OPENAI_INTENT,
  PI_AI_PROVIDER,
  alignCopilotRequest,
  dshMessagesHaveImage,
  inferDshInitiator,
  inspectCopilotWireBody,
  installCopilotWireAlignment,
  isCopilotUrl,
} from '../lib/index.js'

const DSH_UA = 'deepseek-harness/0.2.1-rc.5 (+https://github.com/deepseek-ai/deepseek-harness)'

// ── DSH semantic initiator classification ────────────────────────────────────

test('inferDshInitiator classifies a human-authored initial prompt as user', () => {
  assert.equal(inferDshInitiator({
    provider: PI_AI_PROVIDER,
    model: 'gpt-5.4',
    messages: [
      { role: 'system', content: [{ type: 'text', text: 'sys' }], source: { kind: 'system-prompt' } },
      { role: 'user', content: [{ type: 'text', text: 'hello' }], source: { kind: 'user' } },
      {
        role: 'user',
        content: [{ type: 'text', text: 'Current runtime context: cwd=/repo' }],
        source: { kind: 'runtime-context', form: 'snapshot', sections: [] },
      },
    ],
  }), 'user')
})

test('inferDshInitiator classifies auxiliary purposes (session-title, compaction, auto-review) as agent', () => {
  for (const purpose of ['session-title', 'compaction', 'auto-review']) {
    assert.equal(inferDshInitiator({
      provider: PI_AI_PROVIDER,
      model: 'gpt-5.4',
      purpose,
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'summarize this turn' }], source: { kind: 'user' } },
      ],
    }), 'agent', purpose)
  }
})

test('inferDshInitiator keeps X-Initiator as agent when DSH injects runtime-context or time-context after a tool result', () => {
  assert.equal(inferDshInitiator({
    provider: PI_AI_PROVIDER,
    model: 'claude-sonnet-4.6',
    messages: [
      { role: 'user', content: [{ type: 'text', text: 'fix bug' }], source: { kind: 'user' } },
      {
        role: 'assistant',
        content: [{ type: 'tool-call', id: 'c1', name: 'bash', arguments: '{}' }],
        source: { kind: 'model', provider: PI_AI_PROVIDER, model: 'claude-sonnet-4.6' },
      },
      {
        role: 'tool',
        toolCallId: 'c1',
        content: [{ type: 'text', text: 'ok' }],
        source: { kind: 'tool', callId: 'c1' },
      },
      {
        role: 'user',
        content: [{ type: 'text', text: 'Current runtime context: cwd=/repo/sub' }],
        source: { kind: 'runtime-context', form: 'snapshot', sections: [] },
      },
      {
        role: 'user',
        content: [{ type: 'text', text: 'Time sampled while preparing turn 1, step 2: 2026-11-03' }],
        source: { kind: 'time-context', form: 'snapshot', sections: [] },
      },
      {
        role: 'user',
        content: [{ type: 'text', text: 'Loaded skill instructions...' }],
        source: { kind: 'skill', form: 'instructions' },
      },
    ],
  }), 'agent')
})

test('inferDshInitiator classifies a genuine second human turn after prior tool activity as user', () => {
  assert.equal(inferDshInitiator({
    provider: PI_AI_PROVIDER,
    model: 'gpt-5.4',
    messages: [
      { role: 'user', content: [{ type: 'text', text: 'first question' }], source: { kind: 'user' } },
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'first answer' }],
        source: { kind: 'model', provider: PI_AI_PROVIDER, model: 'gpt-5.4' },
      },
      { role: 'user', content: [{ type: 'text', text: 'second question' }], source: { kind: 'user' } },
      {
        role: 'user',
        content: [{ type: 'text', text: 'Current runtime context: cwd=/repo' }],
        source: { kind: 'runtime-context' },
      },
    ],
  }), 'user')
})

// ── Wire body inspection across all 3 Copilot protocols ──────────────────────

test('inspectCopilotWireBody detects user vs agent and vision on OpenAI Responses (/responses)', () => {
  const userBody = JSON.stringify({
    model: 'gpt-5.5',
    input: [{ role: 'user', content: [{ type: 'input_text', text: 'hi' }] }],
  })
  assert.deepEqual(inspectCopilotWireBody('https://api.individual.githubcopilot.com/responses', userBody), {
    isInference: true,
    agent: false,
    vision: false,
  })

  const toolFollowupWithRuntimeCtx = JSON.stringify({
    model: 'gpt-5.5',
    input: [
      { role: 'user', content: [{ type: 'input_text', text: 'hi' }] },
      { type: 'function_call', call_id: 'c1', name: 'ls', arguments: '{}' },
      { type: 'function_call_output', call_id: 'c1', output: 'file.txt' },
      { role: 'user', content: [{ type: 'input_text', text: 'Current runtime context: none.' }] },
    ],
  })
  assert.deepEqual(inspectCopilotWireBody('https://api.individual.githubcopilot.com/responses', toolFollowupWithRuntimeCtx), {
    isInference: true,
    agent: true,
    vision: false,
  })

  const visionBody = JSON.stringify({
    model: 'gpt-5.5',
    input: [
      {
        role: 'user',
        content: [
          { type: 'input_text', text: 'look' },
          { type: 'input_image', image_url: 'data:image/png;base64,AAA' },
        ],
      },
    ],
  })
  assert.deepEqual(inspectCopilotWireBody('https://api.individual.githubcopilot.com/responses', visionBody), {
    isInference: true,
    agent: false,
    vision: true,
  })
})

test('inspectCopilotWireBody detects user vs agent and vision on Chat Completions (/chat/completions)', () => {
  const agentBody = JSON.stringify({
    model: 'gemini-3.5-flash',
    messages: [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'ls', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: 'c1', content: 'a' },
      { role: 'user', content: 'Time sampled while preparing turn 1, step 2: 2026-11-03' },
    ],
  })
  assert.deepEqual(inspectCopilotWireBody('https://api.individual.githubcopilot.com/chat/completions', agentBody), {
    isInference: true,
    agent: true,
    vision: false,
  })

  const visionBody = JSON.stringify({
    model: 'gemini-3.5-flash',
    messages: [
      {
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AAA' } }],
      },
    ],
  })
  assert.deepEqual(inspectCopilotWireBody('https://api.individual.githubcopilot.com/chat/completions', visionBody), {
    isInference: true,
    agent: false,
    vision: true,
  })
})

test('inspectCopilotWireBody detects tool_result and nested images on Anthropic Messages (/v1/messages)', () => {
  const toolResultWithImage = JSON.stringify({
    model: 'claude-sonnet-4.6',
    messages: [
      { role: 'user', content: 'read screenshot' },
      { role: 'assistant', content: [{ type: 'tool_use', id: 'c1', name: 'read_image', input: {} }] },
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'c1',
            content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AAA' } }],
          },
        ],
      },
    ],
  })
  assert.deepEqual(inspectCopilotWireBody('https://api.individual.githubcopilot.com/v1/messages?beta=true', toolResultWithImage), {
    isInference: true,
    agent: true,
    vision: true,
  })
})

// ── Header shaping and OpenCode alignment ────────────────────────────────────

test('alignCopilotRequest replaces deepseek-harness User-Agent and strips SDK telemetry headers', () => {
  const aligned = alignCopilotRequest('https://api.individual.githubcopilot.com/v1/messages?beta=true', {
    method: 'POST',
    headers: {
      'User-Agent': EXCHANGE_USER_AGENT,
      'user-agent': DSH_UA,
      'Editor-Version': 'vscode/1.107.0',
      'Editor-Plugin-Version': 'copilot-chat/0.35.0',
      'Copilot-Integration-Id': 'vscode-chat',
      'anthropic-dangerous-direct-browser-access': 'true',
      'x-stainless-lang': 'js',
      'x-stainless-os': 'Windows',
      'session_id': 'sess-123',
      'x-client-request-id': 'sess-123',
      'x-api-key': 'tid=123',
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4.6',
      messages: [{ role: 'user', content: 'hello' }],
    }),
  })
  assert.ok(aligned)
  const headers = aligned.init.headers
  assert.equal(headers.get('user-agent'), EXCHANGE_USER_AGENT)
  assert.equal(headers.get('editor-version'), COPILOT_HEADERS['Editor-Version'])
  assert.equal(headers.get('editor-plugin-version'), COPILOT_HEADERS['Editor-Plugin-Version'])
  assert.equal(headers.get('copilot-integration-id'), COPILOT_HEADERS['Copilot-Integration-Id'])
  assert.equal(headers.get('openai-intent'), OPENAI_INTENT)
  assert.equal(headers.get('x-initiator'), 'user')
  assert.equal(headers.get('authorization'), 'Bearer tid=123')
  assert.equal(headers.get('anthropic-version'), '2023-06-01')
  assert.equal(headers.has('x-api-key'), false)
  assert.equal(headers.has('anthropic-dangerous-direct-browser-access'), false)
  assert.equal(headers.has('x-stainless-lang'), false)
  assert.equal(headers.has('x-stainless-os'), false)
  assert.equal(headers.has('session_id'), false)
  assert.equal(headers.has('x-client-request-id'), false)
})

test('alignCopilotRequest does not add inference headers to /models or /copilot_internal/v2/token', () => {
  const activeStore = { initiator: 'user', hasImages: true }
  const tokenReq = alignCopilotRequest('https://api.github.com/copilot_internal/v2/token', {
    method: 'GET',
    headers: { 'user-agent': DSH_UA, Authorization: 'Bearer ghu_xxx' },
  }, activeStore)
  assert.ok(tokenReq)
  assert.equal(tokenReq.init.headers.get('user-agent'), EXCHANGE_USER_AGENT)
  assert.equal(tokenReq.init.headers.has('x-initiator'), false)
  assert.equal(tokenReq.init.headers.has('openai-intent'), false)
  assert.equal(tokenReq.init.headers.has('copilot-vision-request'), false)

  const modelsReq = alignCopilotRequest('https://api.individual.githubcopilot.com/models', {
    method: 'GET',
    headers: { 'user-agent': DSH_UA, Authorization: 'Bearer tid=123' },
  }, activeStore)
  assert.ok(modelsReq)
  assert.equal(modelsReq.init.headers.get('user-agent'), EXCHANGE_USER_AGENT)
  assert.equal(modelsReq.init.headers.has('x-initiator'), false)
  assert.equal(modelsReq.init.headers.has('openai-intent'), false)

  assert.equal(isCopilotUrl('https://api.deepseek.com/v1/chat/completions'), false)
  assert.equal(alignCopilotRequest('https://api.deepseek.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'user-agent': DSH_UA },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
  }, undefined), undefined)
})

test('installCopilotWireAlignment propagates llm/stream context to globalThis.fetch and cleans up on dispose', async () => {
  const seen = []
  const originalFetch = globalThis.fetch
  const baseFetch = async (input, init) => {
    seen.push({ url: String(input), headers: new Headers(init?.headers) })
    return Response.json({ ok: true })
  }
  globalThis.fetch = baseFetch
  let streamWaterfall
  let disposeEffect
  try {
    installCopilotWireAlignment({
      on(event, listener) {
        if (event === 'llm/stream') streamWaterfall = listener
      },
      effect(setup) {
        disposeEffect = setup()
      },
    })
    assert.equal(typeof streamWaterfall, 'function')
    assert.notEqual(globalThis.fetch, baseFetch)

    // Simulate a DSH session-title request where pi-ai would otherwise send X-Initiator: user
    const options = {
      provider: PI_AI_PROVIDER,
      model: 'gpt-5.5',
      purpose: 'session-title',
      messages: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Generate a title' }],
          source: { kind: 'dsh-session-title-llm' },
        },
      ],
    }
    const iterable = streamWaterfall(options, () => ({
      async *[Symbol.asyncIterator]() {
        await globalThis.fetch('https://api.individual.githubcopilot.com/responses', {
          method: 'POST',
          headers: {
            'User-Agent': EXCHANGE_USER_AGENT,
            'user-agent': DSH_UA,
            'X-Initiator': 'user',
          },
          body: JSON.stringify({
            model: 'gpt-5.5',
            input: [{ role: 'user', content: [{ type: 'input_text', text: 'Generate a title' }] }],
          }),
        })
        yield { type: 'finish', reason: { kind: 'stop' } }
      },
    }))
    for await (const _chunk of iterable) {
      // consume stream
    }

    assert.equal(seen.length, 1)
    assert.equal(seen[0].headers.get('user-agent'), EXCHANGE_USER_AGENT)
    assert.equal(seen[0].headers.get('x-initiator'), 'agent')
    assert.equal(seen[0].headers.get('openai-intent'), OPENAI_INTENT)
    assert.equal(dshMessagesHaveImage([
      { role: 'user', content: [{ type: 'image', attachment: { attachmentId: 'a1' } }] },
    ]), true)
    assert.equal(dshMessagesHaveImage([
      { role: 'user', content: [{ type: 'image', offloaded: true, attachment: { attachmentId: 'a1' } }] },
    ]), false)
  } finally {
    disposeEffect?.()
    assert.equal(globalThis.fetch, baseFetch)
    globalThis.fetch = originalFetch
  }
})
