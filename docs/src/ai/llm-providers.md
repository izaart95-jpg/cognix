---
title: LLM Providers - Zed
description: Choose how Zed gets language models: Zed-hosted models, API access, subscriptions, gateways, or local models.
---

# LLM Providers

Use this page to choose which models power [the Zed Agent](./zed-agent.md) and
other Zed-owned AI features, including [Inline Assistant](./inline-assistant.md),
Git commit generation, thread summaries, and similar model-backed features.

Model access paths do not configure [External Agents](./external-agents.md) or
[Terminal Threads](./terminal-threads.md). External Agents and Terminal Threads
usually own their own model access, auth, and configuration.

## Choose a Model Access Path {#choose-a-model-access-path}

| Model access path                                                 | Best when                                                             | Source of truth                       |
| ----------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------- |
| [Use Zed-Hosted Models](../account/zed-hosted-models.md)          | You want models billed through Zed                                    | Account & Billing > Zed-Hosted Models |
| [Use API Access](./use-api-access.md)                             | You have provider API access, credits, or usage billing               | Use API Access                        |
| [Use an Existing Subscription](./use-an-existing-subscription.md) | You already pay for ChatGPT, Claude, Copilot, or another subscription | Use an Existing Subscription          |
| [Use a Gateway](./use-a-gateway.md)                               | You route through OpenRouter, Bedrock, Vercel, or a similar platform  | Use a Gateway                         |
| [Use a Local Model](./use-a-local-model.md)                       | You run models locally or self-hosted                                 | Use a Local Model                     |

Use the setup pages for provider-specific details. See [Agents](./agents.md) for
the difference between the Zed Agent, External Agents, and Terminal Threads.

## Edit Prediction {#edit-prediction}

[Edit Prediction](./edit-prediction.md) has its own provider setup under `edit_predictions`. LLM providers on this page apply to model-backed Zed AI features such as Zed Agent, Inline Assistant, Git commit generation, and thread summaries.

## Anthropic-Compatible Providers {#anthropic-api-compatible}

Anthropic-compatible provider setup has moved to [Use API Access](./use-api-access.md#anthropic-compatible).

## OpenAI-Compatible Providers {#openai-api-compatible}

OpenAI-compatible provider setup has moved to [Use API Access](./use-api-access.md#openai-compatible).

## Server-Managed Custom Providers {#server-managed-custom-providers}

Custom OpenAI-compatible providers (GLM, NIM, TokenRouter, JustWoker, Next
Router) are managed server-side. The app fetches a manifest from
`https://api.cognix.sryze.cc/providers.json` and registers every provider it
describes, so new providers appear without an app update.

Change the manifest URL or refresh cadence in settings:

```json [settings]
{
  "language_models": {
    "custom_providers": {
      "url": "https://api.cognix.sryze.cc/providers.json",
      "refresh_interval_minutes": 10
    }
  }
}
```

`refresh_interval_minutes` defaults to `10`. Set it to `0` to fetch only at
startup and when the URL changes. The last successful manifest is cached on
disk, so an offline start still registers the providers it last knew about.

Each entry in the manifest describes one provider:

```json
{
  "providers": {
    "glm": {
      "active": true,
      "baseUrl": "https://glm.cognix.sryze.cc/v1",
      "apiKey": "sk-example",
      "name": "Cognix-GLM",
      "models": [
        "https://glm.cognix.sryze.cc/v1/models",
        {
          "hardcoded": {
            "id": "glm-5.2",
            "name": "GLM 5.2",
            "input": ["text"],
            "contextWindow": 131072,
            "tools": true,
            "thinking": true,
            "reasoningEfforts": ["high", "max"],
            "defaultReasoningEffort": "high"
          }
        }
      ]
    }
  }
}
```

- The key becomes the provider id (`glm` becomes `cognix.glm`) and the env
  var name for its API key (`GLM_API_KEY`). Keys must not contain `/`, since
  model references are written `provider_id/model_id`. A key stored in the
  system keychain or env var always wins over the manifest's `apiKey`.
- `name` is the display name. It defaults to the key, title-cased.
- `active: false` removes the provider.
- `models` entries are tried in order: URLs first, first success wins.
  Hardcoded entries serve as the fallback when every URL fails.
- Model-list URLs must return an OpenAI-style `{"data": [{"id": ...}]}`
  document. Entries with a `type` other than `"text"` are skipped. Optional
  `reasoning_efforts` and `default_reasoning_effort` fields per entry map to
  the model's reasoning-effort levels.
- The chat-completions endpoint comes from `baseUrl`: a URL already ending
  in `/chat/completions` is used as-is, one ending in a version segment (like
  `/v1`) gets `/chat/completions` appended, otherwise `/v1/chat/completions`
  is appended.
