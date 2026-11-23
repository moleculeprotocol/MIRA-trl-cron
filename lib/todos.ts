import { randomUUID } from "node:crypto"
import type { TodoItem } from "./llm.js"

/** A todo as stored in Sanity (`todoItem` object in `onChainLab.todos`). */
export interface StoredTodo {
  _type: "todoItem"
  _key: string
  criterionKey?: string
  title?: string
  description?: string
  effort?: TodoItem["effort"]
  specificity?: string
  type?: TodoItem["type"]
  status?: "open" | "closed"
}

/** Todos the team marked as closed in Sanity. These survive every rerun. */
export function closedTodosOf(
  doc: Record<string, unknown> | null | undefined,
): StoredTodo[] {
  const todos = doc?.todos
  if (!Array.isArray(todos)) return []
  return todos.filter(
    (todo): todo is StoredTodo =>
      todo?._type === "todoItem" &&
      typeof todo._key === "string" &&
      todo.status === "closed",
  )
}

/**
 * Builds the new `todos` array: freshly generated todos (open) first, then the
 * closed todos kept as they are. New todos get random keys so they can never
 * collide with the keys of closed todos.
 */
export function mergeTodos(
  generated: TodoItem[],
  closed: StoredTodo[],
): StoredTodo[] {
  const open = generated.map(
    (todo): StoredTodo => ({
      _type: "todoItem",
      _key: randomUUID(),
      criterionKey: todo.criterion_key,
      title: todo.title,
      description: todo.description,
      effort: todo.effort,
      specificity: todo.specificity,
      type: todo.type,
      status: "open",
    }),
  )
  return [...open, ...closed]
}
