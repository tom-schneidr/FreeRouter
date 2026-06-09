import { describe, expect, it, vi } from "vitest";
import {
  compareModels,
  filterModels,
  loadSortPreference,
  saveSortPreference,
  toggleSort,
} from "./usagePageLogic";

describe("usagePageLogic", () => {
  it("round-trips sort preference", () => {
    const storage = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => {
        storage.set(key, value);
      },
    });
    saveSortPreference({ key: "tokens", dir: "desc" });
    expect(loadSortPreference()).toEqual({ key: "tokens", dir: "desc" });
  });

  it("toggles sort direction", () => {
    expect(toggleSort({ key: "rank", dir: "asc" }, "rank")).toEqual({ key: "rank", dir: "desc" });
    expect(toggleSort({ key: "rank", dir: "asc" }, "tokens")).toEqual({ key: "tokens", dir: "desc" });
  });

  it("filters by provider and health", () => {
    const models = [
      {
        route_id: "a",
        provider_name: "groq",
        display_name: "A",
        model_id: "a",
        health: { status: "active" },
      },
      {
        route_id: "b",
        provider_name: "nvidia",
        display_name: "B",
        model_id: "b",
        health: { status: "rate_limited" },
      },
    ] as never[];
    expect(filterModels(models, "", "groq", "").map((m) => m.route_id)).toEqual(["a"]);
    expect(filterModels(models, "", "", "rate_limited").map((m) => m.route_id)).toEqual(["b"]);
  });

  it("sorts by rank ascending by default", () => {
    const models = [
      { route_id: "b", rank: 2, provider_name: "x", display_name: "b", model_id: "b" },
      { route_id: "a", rank: 1, provider_name: "x", display_name: "a", model_id: "a" },
    ] as never[];
    expect(compareModels(models[0], models[1], { key: "rank", dir: "asc" })).toBeGreaterThan(0);
  });
});
