// The conversation store, as this app sees it: the wire types for
// `/v1/conversations`, and the two pure translations between the
// server's message tree and assistant-ui's exported repository.
//
// The translations live here rather than in the Chat screen because
// they are the part worth testing without a browser: "which nodes have
// not been stored yet" is where a sync loop either duplicates a message
// or silently drops one, and both failures are invisible until someone
// reloads the page a day later.
//
// The server's shape is deliberately neutral -- role, text, parent --
// so a transcript is readable by something that is not this app.
// Everything assistant-ui needs and the server does not understand
// rides in `metadata`, which the server stores opaquely and hands back
// byte-identical.

import { getJson, postJson, routes } from "./api.ts";
import { readThought, THOUGHT_KEY, type Thought } from "./thought.ts";

export type ConversationRole = "user" | "assistant" | "system";

/** One node of the stored tree. */
export type StoredMessage = {
  id: string;
  parent_id: string | null;
  role: ConversationRole;
  content: string;
  /**
   * A reasoning model's chain of thought, beside the answer rather
   * than inside it. Absent on records written before the store had the
   * field, and on every turn that did not think.
   */
  reasoning_content?: string | null;
  /**
   * How long the model thought, in milliseconds, measured by the
   * client between the first reasoning delta and the first content
   * delta. Beside `reasoning_content` and never without it. Absent on
   * records from before it was stored, which then show their thought
   * with no time.
   */
  reasoning_ms?: number | null;
  created_at: number;
  metadata?: Record<string, unknown> | null;
};

export type Conversation = {
  object: "conversation";
  id: string;
  title: string | null;
  model: string | null;
  created_at: number;
  updated_at: number;
  head_id: string | null;
  messages: StoredMessage[];
};

/**
 * A listed conversation.
 *
 * Tagged differently from a full one by the server on purpose: it
 * carries no `messages`, and a client that treated the two as the same
 * shape would render an empty transcript and believe it.
 */
export type ConversationSummary = {
  object: "conversation.summary";
  id: string;
  title: string | null;
  model: string | null;
  created_at: number;
  updated_at: number;
  head_id: string | null;
  message_count: number;
};

/** A message on the way to the server. It states no timestamp: the
 * server stamps that, and a client-supplied one would be a claim. */
export type NewMessage = {
  id: string;
  parent_id: string | null;
  role: ConversationRole;
  content: string;
  reasoning_content?: string;
  reasoning_ms?: number;
  metadata?: Record<string, unknown>;
};

export type CreateBody = {
  title?: string;
  model?: string;
  head_id?: string;
  messages?: NewMessage[];
};

export type UpdateBody = {
  title?: string;
  model?: string;
  head_id?: string;
  append?: NewMessage[];
};

// ---------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------

export async function listConversations(): Promise<ConversationSummary[]> {
  const body = await getJson<{ data?: ConversationSummary[] }>(
    routes.conversations,
  );
  return body?.data ?? [];
}

export function createConversation(body: CreateBody): Promise<Conversation> {
  return postJson<Conversation>(routes.conversations, body);
}

export function getConversation(id: string): Promise<Conversation> {
  return getJson<Conversation>(routes.conversation(id));
}

export function updateConversation(
  id: string,
  body: UpdateBody,
): Promise<Conversation> {
  return postJson<Conversation>(routes.conversation(id), body);
}

export function deleteConversation(id: string): Promise<unknown> {
  return postJson(routes.conversationDelete(id), {});
}

// ---------------------------------------------------------------------
// assistant-ui <-> the store
// ---------------------------------------------------------------------

/**
 * The parts of an exported node this module reads.
 *
 * Structural rather than imported from `@assistant-ui/react` so that
 * this file stays importable by node's test runner without pulling a
 * React tree in behind it. The real type is wider; nothing here needs
 * the rest of it.
 */
export type ExportedItem = {
  parentId: string | null;
  message: {
    id?: string;
    role: string;
    content: readonly { type: string; text?: string }[];
    status?: { type: string; reason?: string };
    metadata?: Record<string, unknown>;
  };
};

export type ExportedRepository = {
  headId?: string | null;
  messages: readonly ExportedItem[];
};

const ROLES: ReadonlySet<string> = new Set(["user", "assistant", "system"]);

/** Parts of one kind, concatenated. */
function textOfKind(
  content: readonly { type: string; text?: string }[],
  kind: "text" | "reasoning",
): string {
  return content
    .filter((part) => part.type === kind && typeof part.text === "string")
    .map((part) => part.text as string)
    .join("");
}

/** Text parts, concatenated. Anything else in the message is not text
 * and is not what a transcript stores as the answer. */
export function plainText(
  content: readonly { type: string; text?: string }[],
): string {
  return textOfKind(content, "text");
}

/**
 * Reasoning parts, concatenated.
 *
 * Stored in its own field, never folded into the answer. An R1 turn cut
 * off inside its thinking is ALL reasoning; stored as `content: ""` it
 * came back from a reload as an empty bubble with the thinking gone,
 * which is the defect this exists to close.
 */
export function plainReasoning(
  content: readonly { type: string; text?: string }[],
): string {
  return textOfKind(content, "reasoning");
}

/**
 * Whether a node is finished enough to store.
 *
 * A message still streaming is skipped, not stored-then-updated: the
 * server's append is an append, ids are unique, and a half-decoded
 * answer written now would have to be rewritten on every token. It is
 * picked up on the tick after the stream ends.
 *
 * A failed answer is skipped for good. There is nothing in it worth
 * replaying to the model, and `toWire` in the runtime already refuses
 * to send one, so storing it would put a message in the transcript that
 * the next turn silently drops.
 */
export function isStorable(message: ExportedItem["message"]): boolean {
  if (!message.id || !ROLES.has(message.role)) return false;
  const status = message.status?.type;
  if (status === "running" || status === "requires-action") return false;
  if (status === "incomplete" && message.status?.reason === "error")
    return false;
  return true;
}

export type Pending = {
  /** Nodes not yet on the server, parents before children. */
  messages: NewMessage[];
  /**
   * The selected leaf, once it is a node the server has. `null` when
   * the head is a message that was skipped -- sending it would be
   * refused, and a refusal for something the user cannot see is worse
   * than a head that catches up on the next write.
   */
  headId: string | null;
};

/**
 * What still has to be written, given what has already been.
 *
 * A node is included only when its parent is already stored or is
 * included earlier in the same batch, so the server's
 * parent-must-exist rule is satisfied by construction rather than by a
 * retry. A node whose parent was skipped is skipped too, and comes back
 * on the tick after its parent lands.
 */
export function pendingAppend(
  exported: ExportedRepository,
  stored: ReadonlySet<string>,
): Pending {
  const messages: NewMessage[] = [];
  const reachable = new Set<string>(stored);

  for (const item of exported.messages) {
    const { message } = item;
    if (!message.id || reachable.has(message.id)) continue;
    if (!isStorable(message)) continue;
    const parentId = item.parentId;
    if (parentId !== null && !reachable.has(parentId)) continue;
    reachable.add(message.id);
    const reasoning = plainReasoning(message.content);
    const { metadata, thoughtMs } = liftThought(message.metadata);
    messages.push({
      id: message.id,
      parent_id: parentId,
      role: message.role as ConversationRole,
      content: plainText(message.content),
      ...(reasoning ? { reasoning_content: reasoning } : {}),
      ...(reasoning && thoughtMs !== undefined ? { reasoning_ms: thoughtMs } : {}),
      ...(metadata ? { metadata } : {}),
    });
  }

  const head = exported.headId ?? null;
  return { messages, headId: head && reachable.has(head) ? head : null };
}

/**
 * Whether anything in `pending` is worth a request.
 *
 * A withheld head (`null`) is not work. Counting it would spin: there
 * is nothing to send that would change it, so the next export would
 * report the same difference, and the loop would write forever without
 * ever agreeing with itself.
 */
export function hasWork(pending: Pending, storedHead: string | null): boolean {
  if (pending.messages.length > 0) return true;
  return pending.headId !== null && pending.headId !== storedHead;
}

/**
 * The thought's duration, taken OUT of the metadata it rides on.
 *
 * The store keeps `metadata` byte-identical, so leaving
 * `custom.thought` in it would store the duration twice -- once as
 * the typed `reasoning_ms` column and once inside an opaque blob --
 * and a reload would then have two numbers to choose between. It is
 * stored once, as the column, and `toBranchable` puts it back under
 * the same key. A thought still marked as running is not a duration
 * and is dropped rather than stored.
 */
function liftThought(metadata: Record<string, unknown> | undefined): {
  metadata: Record<string, unknown> | undefined;
  thoughtMs: number | undefined;
} {
  const thought = metadata && readThought(metadata);
  if (!metadata || !thought) return { metadata, thoughtMs: undefined };
  const { [THOUGHT_KEY]: _thought, ...custom } = metadata.custom as Record<
    string,
    unknown
  >;
  return {
    metadata: { ...metadata, custom },
    thoughtMs: thought.state === "done" ? thought.ms : undefined,
  };
}

/** The status an assistant node gets back, from the outcome its own
 * metadata recorded. */
export type RestoredStatus =
  | { type: "complete"; reason: "stop" }
  | { type: "incomplete"; reason: "length" | "cancelled" };

/**
 * `status` is reconstructed rather than stored: the store has no
 * business keeping a copy of a fact that is already in the metadata it
 * carries. The outcome is what the runtime wrote under
 * `metadata.custom.stats`, and it is what decides whether a reloaded
 * answer offers Continue -- a cut-off turn restored as complete would
 * lose the way out of it.
 */
export function restoredStatus(
  metadata: Record<string, unknown> | null | undefined,
): RestoredStatus {
  const outcome = (
    metadata as { custom?: { stats?: { outcome?: unknown } } } | undefined
  )?.custom?.stats?.outcome;
  if (outcome === "length") return { type: "incomplete", reason: "length" };
  if (outcome === "stopped-by-you" || outcome === "stopped-by-server")
    return { type: "incomplete", reason: "cancelled" };
  return { type: "complete", reason: "stop" };
}

/**
 * The metadata a restored node carries: what was stored, with the
 * thought's duration put back where the runtime writes it, so the
 * Thinking block reads one shape whether the turn is fresh or
 * reloaded. `null` when there is nothing to carry.
 */
export function restoredMetadata(
  node: Pick<StoredMessage, "metadata" | "reasoning_content" | "reasoning_ms">,
): Record<string, unknown> | null {
  const ms = node.reasoning_ms;
  const thought: Thought | undefined =
    node.reasoning_content && typeof ms === "number" && Number.isFinite(ms)
      ? { state: "done", ms: Math.max(0, ms) }
      : undefined;
  if (!thought) return node.metadata ?? null;
  const custom = node.metadata?.custom;
  return {
    ...node.metadata,
    custom: {
      ...(custom && typeof custom === "object" ? custom : {}),
      [THOUGHT_KEY]: thought,
    },
  };
}

/**
 * The stored tree, in the shape `ExportedMessageRepository
 * .fromBranchableArray` takes.
 *
 * Thinking comes back as a reasoning part ABOVE the text, the order it
 * was shown in; a turn that never thought grows no empty part.
 */
export function toBranchable(conversation: Conversation): {
  items: {
    parentId: string | null;
    message: {
      id: string;
      role: ConversationRole;
      content: (
        | { type: "text"; text: string }
        | { type: "reasoning"; text: string }
      )[];
      createdAt: Date;
      status?: RestoredStatus;
      metadata?: Record<string, unknown>;
    };
  }[];
  headId: string | null;
} {
  return {
    items: conversation.messages.map((node) => {
      const metadata = restoredMetadata(node);
      return {
        parentId: node.parent_id,
        message: {
          id: node.id,
          role: node.role,
          content: [
            ...(node.reasoning_content
              ? [{ type: "reasoning" as const, text: node.reasoning_content }]
              : []),
            { type: "text" as const, text: node.content },
          ],
          createdAt: new Date(node.created_at * 1000),
          ...(node.role === "assistant"
            ? { status: restoredStatus(node.metadata) }
            : {}),
          ...(metadata ? { metadata } : {}),
        },
      };
    }),
    headId: conversation.head_id,
  };
}

/** Ids of every node the server is holding, for the sync loop's
 * "already stored" set. */
export function storedIds(conversation: Conversation): Set<string> {
  return new Set(conversation.messages.map((m) => m.id));
}

/** What to call a conversation with no title yet. */
export function conversationLabel(
  summary: Pick<ConversationSummary, "title" | "id">,
): string {
  const title = summary.title?.trim();
  return title || "Untitled conversation";
}
