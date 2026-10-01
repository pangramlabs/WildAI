---
${front_matter}
---

# WildAI

> ${license_summary}

WildAI is a pool of English web documents from Common Crawl (2024 to mid-2026), each labeled by the Pangram AI-text
detector as human-written, AI-generated or mixed. It was assembled to build pretraining mixtures with a controlled share
of AI-generated text for ${paper}${companions}.

**WildAI is not a natural sample of the web.** An EditLens detector preselected likely-AI documents, so AI text is far
more common in the pool than on the web; `selection` records why each document is in it.${measure_note}

## Composition

${composition}

## Configs

| Config | Contents |
|---|---|
${config_rows}

## How it was built

1. **Collection.** Documents from the Hugging Face FineWeb v1.4.0 dumps CC-MAIN-2024-10 to CC-MAIN-2025-26 (`source =
   fineweb`), and from WARC files of the crawls CC-MAIN-2025-30 to CC-MAIN-2026-25 processed with FineWeb's recipe in
   DataTrove 0.2.0 (`source = common_crawl`).${extension_note}
2. **Preselection.** EditLens (`pangram/editlens_Llama-3.2-3B`, four buckets, mean over up to three 512-token windows)
   labeled the collected documents. Documents in its two AI buckets and a crawl-matched draw from its human bucket
   became candidates (`selection` = `editlens_ai`, `editlens_human`); its lightly-edited bucket was dropped. A
   label-blind random draw from the 2026 crawls was labeled too (`selection = natural`).
3. **Labeling.** Pangram labeled every candidate as Human, Mixed or AI (`pangram_label`), with the fractions of the text
   classified AI, AI-assisted and human. ${pangram_note}
4. **Topics and formats.** WebOrganizer's topic and format classifiers (`WebOrganizer/TopicClassifier`,
   `WebOrganizer/FormatClassifier`) labeled each document.
5. **Pools.** Documents were split by Pangram label and deduplicated by `id`.

## Fields

`id` is the Common Crawl WARC-Record-ID. `token_count` counts GPT-2 tokens of the released text. `truncated` marks text
shorter than the page's extraction. `warc_path` is empty for documents taken from Hugging Face FineWeb (its `file_path`,
joined on `id`, gives it). Sorting a pool by `sampling_hash` gives its order: any prefix is a uniform sample. `pii_anonymized_after_labeling` marks rows whose e-mail or IP addresses were replaced
at release with FineWeb's placeholders, after the labels were computed on the original text.${windows_note}

## Limitations

- Detector labels are not ground truth: Pangram makes errors, and Mixed documents are ambiguous.
- The pool over-represents AI text by design and covers only the crawl files that were processed.
- Deduplication is by document id; the same text can occur under different ids, across crawls.

## Personal information and removal requests

Text went through FineWeb's anonymization of e-mail and IP addresses; other personal information may remain. To request
removal of content, contact ${contact}.

## License

${license_summary} The full text is in `LICENSE`. ${license_text}

## Citation

```bibtex
${citation}
```
