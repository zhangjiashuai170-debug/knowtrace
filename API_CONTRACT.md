# KnowTrace API 参考

## HTTP接口

JSON error detail via FastAPI. Session cookie after `/api/auth/login`, demo users: `admin/admin-demo` all KBs, `engineer/engineer-demo` engineering, `support/support-demo` support. Demo accounts only, not production identity integration. GET `/api/auth/me` -> `{user:{id,display_name,role,kb_ids}}`; POST login username/password -> same; POST logout.

- GET `/api/health` -> status,version,mode='offline',model_connected=false.
- GET `/api/model` (authenticated) -> provider,model,configured,live_verified,last_success_at,default_answer_mode='extractive',message. Configured is not health proof; live_verified means a previous successful request only. Never returns keys/endpoint credentials.
- GET `/api/retrieval` (authenticated) -> available,configured,modes,model,fingerprint,dimension,default_mode,message. Semantic modes are disabled when no verified local encoder is ready. `model` includes bounded non-secret model/version/pooling/truncation metadata.
- GET `/api/kbs` -> `{items:[...]}`; POST `/api/kbs` name,description (admin).
- GET `/api/documents?kb_id=...` -> `{items:[...]}`.
- POST `/api/documents` multipart file,kb_id,optional source_url -> `{document:...}`.
- DELETE `/api/documents/{id}` -> `{deleted:true}`.
- GET `/api/documents/{id}/source` -> managed file, ACL checked.
- GET `/api/documents/{id}/chunks?page=...` -> `{document:...,items:[...]}`.
- POST `/api/ask` question,kb_ids?(optional),mode,top_k,answer_mode='extractive'|'generated' -> evidence result or opt-in model result. Disabled generated mode returns409, evidence failure skips provider, provider/format/citation failure returns502. Generated `grounding.status=citation_checked` validates allowed IDs/markers only, not semantic support.
- POST `/api/search` same -> `{items:[hits],elapsed_ms}`.
- GET `/api/evaluations/datasets` -> `{items:[{id,name,size,answerable,unanswerable,description}]}`.
- POST `/api/evaluations/run` dataset_id,mode (bm25/tfidf/hybrid/semantic/semantic_hybrid),top_k (1..10) -> `{id,mode,metrics,cases,created_at,duration_ms,scope}`. Metrics recall_at_1,recall_at_3,recall_at_5,mrr,refusal_accuracy,false_refusal_rate,p50_ms,p95_ms,search_p50_ms,search_p95_ms. Cases include search_latency_ms separately from warmed ask latency. No made-up scores or LLM-quality claims.
- GET `/api/evaluations/runs` -> `{items:[...]}`.
- GET `/api/evaluations/runs/{id}` -> full run; `/download` -> JSON.
- POST `/api/demo/reset` is NOT provided; preserve user uploads across restarts. Use explicit seed CLI instead.

Evaluation identity is strengthened beyond the initial filename label: official document IDs must match manifest content SHA-256. Reports include dataset/manifest/index/core-file SHA-256. `/api/evaluations/run` also accepts optional `split` dev/test; no split means all. Added metrics expected_source_use_rate and unsupported_acceptance_rate; per-case passed requires expected source actually used, not merely retrieved. Evaluation never calls model providers. v0.2 test comparison is regression on a previously viewed split, not unseen holdout.

All question/search/evaluation mode fields accept the five modes above. Explicit semantic mode without an encoder returns409; inference/cache failures return a sanitized503, never relabeled lexical results. Vector storage is SQLite exact-cosine search, not an ANN engine. Rows are ACL-filtered before vector loading or ranking; cache keys bind chunk ID, model fingerprint, text SHA and dimension. First semantic search may build document vectors; later query encoding still takes CPU time.

