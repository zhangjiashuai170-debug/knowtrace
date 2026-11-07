"""Lexical and optional local-vector ranking of an already-authorized corpus."""

from collections import Counter
import math
import re
import unicodedata
import logging
import threading

import jieba

STOP = {
    "what", "why", "when", "where", "which", "who", "how", "the", "a", "an", "and",
    "or", "is", "are", "do", "does", "to", "of", "in", "on", "for", "with", "can",
    "could", "please", "tell", "me", "about", "什么", "如何", "怎么", "怎样", "是否",
    "哪些", "为何", "为什么", "一个", "可以", "这里", "这样", "以及", "进行", "使用",
    "需要", "请问", "介绍", "一下", "一下子", "的是", "了什么", "有什么",
    "should", "would", "will", "has", "have", "had", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "their", "our", "your",
    "you", "i", "we", "they", "as", "at", "by", "from", "into", "than", "then",
    "given", "explain", "describe", "show", "give", "which", "must",
    "的", "地", "得", "了", "吗", "呢", "吧", "啊", "与", "和", "或", "及", "也",
    "就", "还", "都", "再", "更", "把", "被", "将", "从", "到", "对", "为", "以",
    "在", "由", "于", "而", "当", "时", "后", "前", "中", "里", "内", "之",
    "我", "我们", "你", "您", "它", "它们", "这个", "那个", "这些", "那些", "某个",
    "一种", "一类", "本身", "应该", "应", "能够", "希望", "请", "告诉", "说明",
    "哪个", "哪种", "哪里", "能否", "可否", "能", "会", "要", "想", "用", "有",
    "哪", "可", "供", "是", "另", "一套",
}

# Separate tokenizer instance: bundled dictionary only, no user dictionaries,
# no network and no changes to jieba's process-global tokenizer.
_CHINESE = jieba.Tokenizer()
jieba.setLogLevel(logging.WARNING)

MINIMUM_COVERAGE = .35
NEGATIONS = {"not", "no", "never", "without", "cannot", "不", "没", "没有", "未", "无",
             "非", "不能", "不可", "不允许", "禁止", "不得", "无需", "不必", "不可能"}

LEXICAL_MODES = ("bm25", "tfidf", "hybrid")
SEMANTIC_MODES = ("semantic", "semantic_hybrid")
RETRIEVAL_MODES = LEXICAL_MODES + SEMANTIC_MODES


def semantic_failure(message):
    # The optional runtime module has no optional-library imports at module
    # scope. Import the public error lazily so lexical-only use stays separate.
    from .embeddings import SemanticError
    return SemanticError(message)


def embedding_identity(embedder) -> tuple[str, int]:
    if embedder is None:
        raise semantic_failure("尚未配置本地语义模型，请先准备模型或选择词法检索。")
    try:
        fingerprint, dimension = embedder.fingerprint, embedder.dimension
        if (not isinstance(fingerprint, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", fingerprint)
                or isinstance(dimension, bool) or not isinstance(dimension, int)
                or not 1 <= dimension <= 4096):
            raise ValueError()
        return fingerprint, dimension
    except Exception:
        raise semantic_failure("本地语义模型配置无效。") from None


def normalized_vector(value, dimension: int, *, cached=False) -> list[float]:
    try:
        if not isinstance(value, (list, tuple)) or len(value) != dimension:
            raise ValueError()
        if any(isinstance(number, bool) or not isinstance(number, (int, float))
               or not math.isfinite(number) for number in value):
            raise ValueError()
        norm = math.hypot(*value)
        if not math.isfinite(norm) or norm <= 0 or (cached and abs(norm - 1) > .001):
            raise ValueError()
        return [float(number / norm) for number in value]
    except (ValueError, TypeError, OverflowError):
        raise semantic_failure("本地语义向量或缓存格式无效，未回退为其他检索结果。") from None


def _technical_terms(question: str) -> set[str]:
    """Preserve out-of-corpus acronyms, mixed-case identifiers and versions."""
    terms = set()
    for value in re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[_.-][A-Za-z0-9]+)*", unicodedata.normalize("NFKC", question)):
        if (re.search(r"[a-z][A-Z]", value) or re.search(r"[0-9_.-]", value)
                or (len(value) >= 2 and value.isupper())):
            terms.update(tokens(value))
    return terms


def _numeric_question(question: str) -> bool:
    # “适合多大的文件” can be grounded by “小文件”; do not demand an
    # invented numeric limit unless an actual quantity is requested.
    return bool(re.search(r"多少|几(?:个|次|天|秒|小时|分钟|字节)|(?:最大|最小|默认|具体|精确).{0,8}(?:多大|多长)|\bhow\s+(?:many|much|long)\b|\bexact\b.*\b(?:size|limit|duration|number)\b",
                          normalize(question)))


def evidence_rejections(question: str, query, counts, vocabulary, text: str) -> list[str]:
    reasons = []
    if NEGATIONS.intersection(query) and not NEGATIONS.intersection(counts):
        reasons.append("missing_explicit_negation")
    numbers = {term for term in query if re.fullmatch(r"\d+(?:[.-]\d+)*", term)}
    if numbers.difference(counts):
        reasons.append("missing_numeric_constraint")
    if _technical_terms(question).difference(vocabulary):
        reasons.append("unknown_technical_identifier")
    if _numeric_question(question):
        lowered = normalize(text)
        qualifiers = {term for term in query if term in {"最大", "最小", "默认", "maximum", "minimum", "default"}}
        if qualifiers.difference(counts):
            reasons.append("missing_numeric_qualifier")
        units = bool(re.search(r"字节|bytes?\b|毫秒|milliseconds?\b|秒|seconds?\b|分钟|minutes?\b|小时|hours?\b", normalize(question)))
        numeric_value = (re.search(r"\d+(?:\.\d+)?\s*(?:字节|bytes?\b|[kmgt]i?b\b|毫秒|ms\b|秒|seconds?\b|分钟|minutes?\b|小时|hours?\b)", lowered)
                         if units else re.search(r"\d+", lowered))
        if not numeric_value:
            reasons.append("numeric_value_not_found")
    return reasons


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower().replace("’", "'")
    value = value.replace("can't", "can not").replace("won't", "will not")
    return re.sub(r"n['’]t\b", " not", value)


def _english(word: str) -> str:
    # Conservative inflection normalization, not a domain synonym expansion.
    if not word.isalpha() or not word.isascii() or len(word) < 4:
        return word
    stem = word
    if stem.endswith(("ies", "ied")) and len(stem) > 4:
        stem = stem[:-3] + "y"
    elif stem.endswith("ing") and len(stem) > 5:
        stem = stem[:-3]
    elif stem.endswith("ed") and len(stem) > 4:
        stem = stem[:-2]
    elif stem.endswith(("sses", "ches", "shes", "xes", "zes")):
        stem = stem[:-2]
    elif stem.endswith("s") and not stem.endswith(("ss", "us", "is")):
        stem = stem[:-1]
    if word.endswith(("ing", "ed")) and len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
        stem = stem[:-1]
    if len(stem) >= 4 and stem.endswith("e"):
        stem = stem[:-1]
    return stem


def tokens(value: str) -> list[str]:
    result = []
    for word in re.findall(r"[a-z0-9]+(?:[_.-][a-z0-9]+)*|[\u3400-\u9fff]+", normalize(value)):
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            values = list(_CHINESE.cut(word, HMM=False))
        else:
            # Header/identifier spelling variants share a term; numeric values
            # keep punctuation so 1.2 is never silently treated as 12.
            canonical = word if word[0].isdigit() else re.sub(r"[_.-]", "", word)
            values = [] if canonical in STOP else [_english(canonical)]
        result.extend(part for part in values if part and part not in STOP)
    return result


def characters(value: str) -> Counter:
    normalized = re.sub(r"[^a-z0-9_\u3400-\u9fff]", "", normalize(value))
    if len(normalized) == 1:
        return Counter({normalized: 1})
    return Counter(normalized[i:i + length] for length in (2, 3, 4)
                   for i in range(max(0, len(normalized) - length + 1)))


class RankIndex:
    def __init__(self, rows: list[dict], *, embedder=None, vector_loader=None):
        self.rows = rows
        self.embedder = embedder
        self._vector_loader = vector_loader
        self._dense_vectors = None
        self._dense_lock = threading.RLock()
        self.word_counts = [Counter(tokens(row["text"])) for row in rows]
        self.lengths = [sum(counts.values()) for counts in self.word_counts]
        self.average_length = sum(self.lengths) / max(1, len(rows))
        self.word_df = Counter(token for counts in self.word_counts for token in counts)
        self.idf = {token: math.log(1 + (len(rows) - count + .5) / (count + .5))
                    for token, count in self.word_df.items()}
        self.unknown_idf = math.log(1 + (len(rows) + .5) / .5)
        feature_counts = [characters(row["text"]) for row in rows]
        feature_df = Counter(feature for counts in feature_counts for feature in counts)
        self.character_idf = {
            feature: math.log((1 + len(rows)) / (1 + count)) + 1
            for feature, count in feature_df.items()
        }
        self.vectors = []
        for counts in feature_counts:
            vector = {feature: (1 + math.log(count)) * self.character_idf[feature]
                      for feature, count in counts.items()}
            norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1
            self.vectors.append({feature: weight / norm for feature, weight in vector.items()})

    def _semantic_scores(self, question: str) -> list[float]:
        _, dimension = embedding_identity(self.embedder)
        with self._dense_lock:
            try:
                if self._dense_vectors is None:
                    values = (self._vector_loader(self.rows) if self._vector_loader is not None
                              else self.embedder.embed_documents([row["text"] for row in self.rows]))
                    if not isinstance(values, list) or len(values) != len(self.rows):
                        raise semantic_failure("本地语义模型返回的文档向量数量无效。")
                    self._dense_vectors = [normalized_vector(value, dimension) for value in values]
                query_vector = normalized_vector(self.embedder.embed_query(question), dimension)
                return [max(-1.0, min(1.0, sum(left * right for left, right in zip(query_vector, value))))
                        for value in self._dense_vectors]
            except Exception as error:
                from .embeddings import SemanticError
                if isinstance(error, SemanticError):
                    raise
                raise SemanticError("本地语义编码未完成，未采用其他模式替代结果。") from None

    def search(self, question: str, mode: str, top_k: int) -> list[dict]:
        if mode not in RETRIEVAL_MODES:
            raise ValueError("检索模式无效")
        semantic = mode in SEMANTIC_MODES
        if semantic:
            embedding_identity(self.embedder)
        query = Counter(tokens(question))
        if not self.rows or (not query and not semantic):
            return []
        query_features = characters(question)
        vector = {feature: (1 + math.log(count)) * self.character_idf[feature]
                  for feature, count in query_features.items() if feature in self.character_idf}
        norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1
        vector = {feature: weight / norm for feature, weight in vector.items()}
        bm25 = []
        tfidf = []
        coverage = []
        mass = sum(self.idf.get(token, self.unknown_idf) for token in query)
        for index, counts in enumerate(self.word_counts):
            score = 0.0
            for token, query_count in query.items():
                count = counts.get(token, 0)
                if count:
                    denominator = count + 1.5 * (.25 + .75 * self.lengths[index] / max(1, self.average_length))
                    score += self.idf[token] * count * 2.5 / denominator * (1 + math.log(query_count))
            bm25.append(score)
            tfidf.append(sum(weight * self.vectors[index].get(feature, 0)
                             for feature, weight in vector.items()))
            found_mass = sum(self.idf[token] for token in query if token in counts)
            coverage.append(found_mass / mass if mass else 0)
        tie = lambda i: (self.rows[i]["document_name"], self.rows[i]["page"] or 0,
                         self.rows[i]["line_start"], self.rows[i]["chunk_id"])
        bm_order = sorted((i for i in range(len(self.rows)) if bm25[i] > 0),
                          key=lambda i: (-bm25[i], tie(i)))
        tf_order = sorted((i for i in range(len(self.rows)) if tfidf[i] >= .03),
                          key=lambda i: (-tfidf[i], tie(i)))
        semantic_scores = self._semantic_scores(question) if semantic else None
        semantic_order = (sorted(range(len(self.rows)), key=lambda i: (-semantic_scores[i], tie(i)))
                          if semantic else [])
        if mode == "bm25":
            order = bm_order
            scores = bm25
        elif mode == "tfidf":
            order = tf_order
            scores = tfidf
        elif mode == "semantic":
            order = semantic_order
            scores = semantic_scores
        else:
            scores = [0.0] * len(self.rows)
            lists = (bm_order, tf_order, semantic_order) if semantic else (bm_order, tf_order)
            for order_list in lists:
                for position, index in enumerate(order_list, 1):
                    scores[index] += 1 / (60 + position)
            order = sorted(set().union(*lists), key=lambda i: (-scores[i], tie(i)))
        hits = []
        for index in order[:top_k]:
            hit = dict(self.rows[index])
            hit.update(score=round(scores[index], 8), bm25_score=round(bm25[index], 6),
                       tfidf_score=round(tfidf[index], 6), coverage=round(coverage[index], 6),
                       query_terms=sorted(query),
                       matched_terms=sorted(token for token in query if token in self.word_counts[index]),
                       missing_terms=sorted(token for token in query if token not in self.word_counts[index]),
                       evidence_rejections=evidence_rejections(
                           question, query, self.word_counts[index], self.word_df, self.rows[index]["text"]))
            if semantic:
                # Cosine is a ranking signal, never an answer-confidence value.
                hit["semantic_score"] = round(semantic_scores[index], 6)
            hits.append(hit)
        return hits
