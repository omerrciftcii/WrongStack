import { describe, expect, it, vi } from 'vitest';
import { OpenAIProvider } from '../src/openai.js';

function mockFetch(json: unknown, status = 200) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => json,
    text: async (): Promise<string> => JSON.stringify(json),
  } as never as Response);
}

describe('OpenAIProvider', () => {
  // Content-parsing tests live in streaming.test.ts since complete() wraps
  // stream() internally. This file covers headers, URLs, errors, and the
  // request-body shape.

  it('non-2xx becomes ProviderError', async () => {
    const fetchImpl = mockFetch({ error: 'auth' }, 401) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await expect(
      p.complete(
        { model: 'm', messages: [{ role: 'user', content: 'x' }], maxTokens: 1 },
        { signal: new AbortController().signal },
      ),
    ).rejects.toMatchObject({ status: 401 });
  });

  it('requires apiKey', () => {
    expect(() => new OpenAIProvider({ apiKey: '' })).toThrow(/apiKey required/);
  });

  it('marks 429 and 5xx as retryable', async () => {
    const fetchImpl = mockFetch({}, 429) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await expect(
      p.complete(
        { model: 'm', messages: [{ role: 'user', content: 'x' }], maxTokens: 1 },
        { signal: new AbortController().signal },
      ),
    ).rejects.toMatchObject({ status: 429, retryable: true });
  });

  it('wraps fetch network failure in ProviderError(retryable)', async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error('ECONNRESET')) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await expect(
      p.complete(
        { model: 'm', messages: [{ role: 'user', content: 'x' }], maxTokens: 1 },
        { signal: new AbortController().signal },
      ),
    ).rejects.toMatchObject({ status: 0, retryable: true });
  });

  it('rethrows abort errors directly', async () => {
    const ctrl = new AbortController();
    const fetchImpl = vi.fn().mockImplementation(async () => {
      ctrl.abort();
      throw new Error('aborted');
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await expect(
      p.complete(
        { model: 'm', messages: [{ role: 'user', content: 'x' }], maxTokens: 1 },
        { signal: ctrl.signal },
      ),
    ).rejects.toThrow(/aborted/);
  });

  it('includes tool_choice when set to named function', async () => {
    let captured: Record<string, unknown> | undefined;
    let capturedUrl: string | undefined;
    const fetchImpl = vi.fn(async (url: unknown, init: { body?: string } = {}) => {
      capturedUrl = String(url);
      captured = JSON.parse(init.body ?? '{}');
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await p.complete(
      {
        model: 'gpt-4o',
        messages: [{ role: 'user', content: 'x' }],
        maxTokens: 1,
        tools: [
          {
            name: 'read',
            description: '',
            inputSchema: { type: 'object' },
            permission: 'auto',
            mutating: false,
            async execute() {
              return '';
            },
          },
        ],
        toolChoice: { type: 'tool', name: 'read' } as never as 'auto',
      },
      { signal: new AbortController().signal },
    );
    expect(captured?.['tool_choice']).toMatchObject({
      type: 'function',
      function: { name: 'read' },
    });
    expect(capturedUrl).toBe('https://api.openai.com/v1/chat/completions');
  });

  it('caps output via max_completion_tokens, not the deprecated max_tokens (#10)', async () => {
    let captured: Record<string, unknown> | undefined;
    const fetchImpl = vi.fn(async (_url: unknown, init: { body?: string } = {}) => {
      captured = JSON.parse(init.body ?? '{}');
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'gpt-4o',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await p.complete(
      { model: 'gpt-4o', messages: [{ role: 'user', content: 'x' }], maxTokens: 256 },
      { signal: new AbortController().signal },
    );
    expect(captured?.['max_completion_tokens']).toBe(256);
    expect(captured?.['max_tokens']).toBeUndefined();
  });

  it('appends /chat/completions to z.ai-style versioned baseUrl', async () => {
    let calledUrl = '';
    const fetchImpl = vi.fn(async (url: unknown) => {
      calledUrl = String(url);
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({
      apiKey: 'k',
      baseUrl: 'https://api.z.ai/api/coding/paas/v4',
      fetchImpl,
    });
    await p.complete(
      { model: 'glm-4.6', messages: [{ role: 'user', content: 'x' }], maxTokens: 1 },
      { signal: new AbortController().signal },
    );
    expect(calledUrl).toBe('https://api.z.ai/api/coding/paas/v4/chat/completions');
  });

  it('uses baseUrl with /chat/completions already as-is', async () => {
    let calledUrl = '';
    const fetchImpl = vi.fn(async (url: unknown) => {
      calledUrl = String(url);
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({
      apiKey: 'k',
      baseUrl: 'https://example.com/v1/chat/completions',
      fetchImpl,
    });
    await p.complete(
      { model: 'm', messages: [{ role: 'user', content: 'x' }], maxTokens: 1 },
      { signal: new AbortController().signal },
    );
    expect(calledUrl).toBe('https://example.com/v1/chat/completions');
  });

  it('adds organization header when set', async () => {
    const spy = vi.fn(async (_url: unknown, init?: { headers?: Record<string, string> }) => {
      return {
        ok: true,
        status: 200,
        headers: init?.headers,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    });
    const p = new OpenAIProvider({
      apiKey: 'k',
      organization: 'org-x',
      fetchImpl: spy as never as typeof fetch,
    });
    await p.complete(
      { model: 'm', messages: [{ role: 'user', content: 'x' }], maxTokens: 1 },
      { signal: new AbortController().signal },
    );
    expect(
      (spy.mock.calls[0]![1] as { headers: Record<string, string> }).headers['openai-organization'],
    ).toBe('org-x');
  });

  it('sends frequency_penalty, presence_penalty, seed when set', async () => {
    let captured: Record<string, unknown> | undefined;
    const fetchImpl = vi.fn(async (_url: unknown, init: { body?: string } = {}) => {
      captured = JSON.parse(init.body ?? '{}');
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await p.complete(
      {
        model: 'o4',
        messages: [{ role: 'user', content: 'hi' }],
        maxTokens: 100,
        frequencyPenalty: 0.5,
        presencePenalty: 0.3,
        seed: 42,
      },
      { signal: new AbortController().signal },
    );
    expect(captured?.['frequency_penalty']).toBe(0.5);
    expect(captured?.['presence_penalty']).toBe(0.3);
    expect(captured?.['seed']).toBe(42);
  });

  it('sends user when set', async () => {
    let captured: Record<string, unknown> | undefined;
    const fetchImpl = vi.fn(async (_url: unknown, init: { body?: string } = {}) => {
      captured = JSON.parse(init.body ?? '{}');
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await p.complete(
      { model: 'm', messages: [{ role: 'user', content: 'hi' }], maxTokens: 100, user: 'user-abc' },
      { signal: new AbortController().signal },
    );
    expect(captured?.['user']).toBe('user-abc');
  });

  it('sends logprobs and top_logprobs when set', async () => {
    let captured: Record<string, unknown> | undefined;
    const fetchImpl = vi.fn(async (_url: unknown, init: { body?: string } = {}) => {
      captured = JSON.parse(init.body ?? '{}');
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await p.complete(
      {
        model: 'm',
        messages: [{ role: 'user', content: 'hi' }],
        maxTokens: 100,
        logprobs: true,
        topLogprobs: 5,
      },
      { signal: new AbortController().signal },
    );
    expect(captured?.['logprobs']).toBe(true);
    expect(captured?.['top_logprobs']).toBe(5);
  });

  it('sends response_format for json_schema', async () => {
    let captured: Record<string, unknown> | undefined;
    const fetchImpl = vi.fn(async (_url: unknown, init: { body?: string } = {}) => {
      captured = JSON.parse(init.body ?? '{}');
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: '{}' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await p.complete(
      {
        model: 'm',
        messages: [{ role: 'user', content: 'hi' }],
        maxTokens: 100,
        responseFormat: {
          type: 'json_schema',
          jsonSchema: {
            name: 'person',
            schema: { type: 'object', properties: { name: { type: 'string' } } },
          },
        },
      },
      { signal: new AbortController().signal },
    );
    expect(captured?.['response_format']).toEqual({
      type: 'json_schema',
      json_schema: {
        name: 'person',
        strict: true,
        schema: { type: 'object', properties: { name: { type: 'string' } } },
      },
    });
  });

  it('sends reasoning_effort when effort is a valid OpenAI value', async () => {
    let captured: Record<string, unknown> | undefined;
    const fetchImpl = vi.fn(async (_url: unknown, init: { body?: string } = {}) => {
      captured = JSON.parse(init.body ?? '{}');
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    await p.complete(
      {
        model: 'o4',
        messages: [{ role: 'user', content: 'hi' }],
        maxTokens: 100,
        reasoning: { effort: 'high' },
      },
      { signal: new AbortController().signal },
    );
    expect(captured?.['reasoning_effort']).toBe('high');
  });

  it('routes first-party tools plus reasoning through Responses', async () => {
    let captured: Record<string, unknown> | undefined;
    let capturedUrl: string | undefined;
    const fetchImpl = vi.fn(async (url: unknown, init: { body?: string } = {}) => {
      capturedUrl = String(url);
      captured = JSON.parse(init.body ?? '{}');
      return new Response(
        [
          'data: {"type":"response.created","response":{"id":"r1","model":"gpt-5.6-sol"}}',
          'data: {"type":"response.output_text.delta","delta":"ok"}',
          'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}',
          '',
        ].join('\n\n'),
        { status: 200, headers: { 'content-type': 'text/event-stream' } },
      );
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ id: 'chatemr-api', apiKey: 'k', fetchImpl });
    await p.complete(
      {
        model: 'gpt-5.6-sol',
        messages: [{ role: 'user', content: 'hi' }],
        maxTokens: 100,
        reasoning: { effort: 'medium' },
        tools: [
          {
            name: 'read',
            description: 'Read a file',
            inputSchema: { type: 'object', properties: {} },
            permission: 'auto',
            mutating: false,
            async execute() {},
          },
        ],
      },
      { signal: new AbortController().signal },
    );
    expect(p.id).toBe('chatemr-api');
    expect(capturedUrl).toBe('https://api.openai.com/v1/responses');
    expect(captured).toMatchObject({
      model: 'gpt-5.6-sol',
      stream: true,
      input: [{ role: 'user', content: [{ type: 'input_text', text: 'hi' }] }],
      reasoning: { effort: 'medium', summary: 'auto' },
    });
    expect(Array.isArray(captured?.['tools'])).toBe(true);
    expect((captured?.['tools'] as Array<Record<string, unknown>> | undefined)?.[0]).toMatchObject({
      type: 'function',
      name: 'read',
    });
    expect(captured?.['reasoning_effort']).toBeUndefined();
    expect(captured?.['messages']).toBeUndefined();
  });

  it('sends the real wire levels (minimal, xhigh) and learns from a refusal', async () => {
    let captured: Record<string, unknown> | undefined;
    const fetchImpl = vi.fn(async (_url: unknown, init: { body?: string } = {}) => {
      captured = JSON.parse(init.body ?? '{}');
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'm',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', fetchImpl });
    for (const effort of ['minimal', 'xhigh'] as const) {
      await p.complete(
        {
          model: 'gpt-5.2',
          messages: [{ role: 'user', content: 'hi' }],
          maxTokens: 100,
          reasoning: { effort },
        },
        { signal: new AbortController().signal },
      );
      // Both are Chat Completions values; dropping them made the user's pick
      // a silent no-op on every first-party request.
      expect(captured?.['reasoning_effort']).toBe(effort);
    }
  });

  it('retries without reasoning_effort when the model refuses the level, then stops sending it', async () => {
    const bodies: Array<Record<string, unknown>> = [];
    const fetchImpl = vi.fn(async (_url: unknown, init: { body?: string } = {}) => {
      const body = JSON.parse(init.body ?? '{}') as Record<string, unknown>;
      bodies.push(body);
      if (body['reasoning_effort'] !== undefined) {
        return {
          ok: false,
          status: 400,
          text: async (): Promise<string> =>
            JSON.stringify({
              error: {
                message: "Unsupported value: 'reasoning_effort' does not support 'minimal'",
                type: 'invalid_request_error',
              },
            }),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: 'o3',
          choices: [{ message: { role: 'assistant', content: 'k' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        text: async (): Promise<string> => '',
      };
    }) as never as typeof fetch;
    const p = new OpenAIProvider({ apiKey: 'k', id: 'openai-probe', fetchImpl });
    const req = {
      model: 'o3',
      messages: [{ role: 'user' as const, content: 'hi' }],
      maxTokens: 100,
      reasoning: { effort: 'minimal' as const },
    };

    await p.complete(req, { signal: new AbortController().signal });
    expect(bodies).toHaveLength(2);
    expect(bodies[0]?.['reasoning_effort']).toBe('minimal');
    expect(bodies[1]).not.toHaveProperty('reasoning_effort');

    // The lesson is per (provider, model) and holds for the session.
    await p.complete(req, { signal: new AbortController().signal });
    expect(bodies).toHaveLength(3);
    expect(bodies[2]).not.toHaveProperty('reasoning_effort');
  });
});
