import assert from "node:assert/strict"
import { test } from "node:test"
import type { TodoItem } from "./llm.js"
import { closedTodosOf, mergeTodos, type StoredTodo } from "./todos.js"

const stored = (key: string, status?: "open" | "closed"): StoredTodo => ({
  _type: "todoItem",
  _key: key,
  title: key,
  status,
})

const generated: TodoItem = {
  criterion_key: "intellectual_property",
  title: "Upload granted patent",
  description: "…",
  effort: "low",
  specificity: "Patent PDF",
  type: "public",
}

test("closedTodosOf keeps only closed todos", () => {
  const doc = {
    todos: [stored("a", "open"), stored("b", "closed"), stored("c"), null],
  }
  assert.deepEqual(
    closedTodosOf(doc).map((t) => t._key),
    ["b"],
  )
})

test("closedTodosOf handles missing documents and fields", () => {
  assert.deepEqual(closedTodosOf(null), [])
  assert.deepEqual(closedTodosOf({}), [])
  assert.deepEqual(closedTodosOf({ todos: "nope" }), [])
})

test("mergeTodos puts new open todos first and keeps closed todos unchanged", () => {
  const closed = [stored("ip-0", "closed")]
  const merged = mergeTodos([generated, generated], closed)

  assert.equal(merged.length, 3)
  assert.deepEqual(merged[2], closed[0])
  for (const todo of merged.slice(0, 2)) {
    assert.equal(todo.status, "open")
    assert.equal(todo.criterionKey, "intellectual_property")
  }

  const keys = new Set(merged.map((t) => t._key))
  assert.equal(keys.size, merged.length, "keys must be unique")
})
