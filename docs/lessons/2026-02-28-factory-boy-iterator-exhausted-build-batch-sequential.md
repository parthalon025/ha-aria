# Lesson: factory.Iterator Exhausted After First build_batch — Silent Empty Values on Subsequent Factory Calls

**Date:** 2026-02-28
**System:** community (FactoryBoy/factory_boy)
**Tier:** lesson
**Category:** testing
**Keywords:** factory_boy, Iterator, build_batch, StopIteration, factory, sequential, test data, fixture
**Source:** https://github.com/FactoryBoy/factory_boy/issues/1040

---

## Observation (What Happened)
A test setup called `factory.build_batch(25)` sequentially for three different factories (UserFactory, TagFactory, TagCaseFactory) using `map()`. The first factory ran successfully. The second and third raised `StopIteration` when their `factory.Iterator` fields were invoked. Running each factory in isolation worked fine.

## Analysis (Root Cause — 5 Whys)
**Why #1:** `factory.Iterator` wraps a Python iterator object. Once the iterator is exhausted (all values consumed), it raises `StopIteration` on subsequent calls.
**Why #2:** `build_batch(25)` consumed all values in the iterator on the first call. The same `Iterator` instance is shared at the class level across all factory calls in a session.
**Why #3:** `factory.Iterator` does not auto-reset by default — it is a stateful object that cycles only when `cycle=True` is explicitly set.
**Why #4:** When `cycle=True` is not set and the iterator runs out of values, the factory silently fails or raises instead of resetting.
**Why #5:** Multiple sequential factory calls in the same test session share the same iterator state because factories are class-level definitions, not per-test instances.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use `factory.Iterator([...], cycle=True)` for any value list that may be consumed across multiple batch calls or test runs | proposed | community | issue #1040 |
| 2 | For unique sequential values, use `factory.Sequence(lambda n: ...)` instead of `factory.Iterator` — sequences are guaranteed to produce unique values and never exhaust | proposed | community | issue #1040 |
| 3 | Use `factory.Faker(...)` for random-valued fields where uniqueness is not required — avoids iterator exhaustion entirely | proposed | community | issue #1040 |

## Key Takeaway
`factory.Iterator` is a stateful, non-cycling iterator by default — it exhausts after one `build_batch` call and raises `StopIteration` on subsequent factory calls in the same session; always use `factory.Iterator([...], cycle=True)` or `factory.Sequence()` for fields consumed across multiple factory calls.
