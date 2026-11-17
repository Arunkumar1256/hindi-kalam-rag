# Hindi RAG Project — Final Answers

Source: Agni Ki Udaan Hindi Study Guide

## Q1. SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?

**Answer:** SLV-III ने रोहिणी उपग्रह को 1980 में कक्षा में स्थापित किया।

**Citation:**
- Page: 8
- Section: 6 — इसरो के वर्ष और SLV-III
- Chunk ID: `p08-c03-e99e1bb7`

## Q2. कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?

**Answer:** कलाम को भारत रत्न 1997 में प्राप्त हुआ।

**Citation:**
- Page: 16
- Section: 14 — पुरस्कार एवं सम्मान
- Chunk ID: `p16-c01-805277bf`

## Q3. पोखरण-II में कलाम की क्या भूमिका थी?

**Answer:** कलाम उस समय रक्षा मंत्री के वैज्ञानिक सलाहकार तथा DRDO के प्रमुख थे। उन्होंने परमाणु ऊर्जा विभाग के वैज्ञानिकों के साथ मिलकर पोखरण-II में केंद्रीय संगठनात्मक भूमिका निभाई।

**Citation:**
- Page: 10
- Section: 8 — पोखरण-II: 1998 के परमाणु परीक्षण
- Chunk ID: `p10-c01-49e8082a`

## Q4. Which institution did Kalam attend to study aeronautical engineering?

**Answer:** Kalam attended the Madras Institute of Technology (MIT), Chennai, to study aeronautical engineering.

**Citation:**
- Page: 6
- Section: 4 — कॉलेज के वर्ष और उड़ान का आकर्षण
- Chunk ID: `p06-c02-2d647c49`

## Q5. Who co-wrote the autobiography, and in what year was it published?

**Answer:** The autobiography *Wings of Fire* was co-written by scientist Arun Tiwari and published in 1999.

**Citation:**
- Page: 3
- Section: 1 — ए.पी.जे. अब्दुल कलाम कौन थे?
- Chunk ID: `p03-c02-346fb9ca`

## Q6. How and where did Kalam die in 2015?

**Answer:** On 27 July 2015, Kalam collapsed while delivering a lecture to students at the Indian Institute of Management Shillong. He died shortly afterward, aged 83.

**Citation:**
- Page: 15
- Section: 13 — अंतिम क्षण तक शिक्षक (2007–2015)
- Chunk ID: `p15-c02-23fd703e`

---

## Implementation summary

- PDF extraction: PyMuPDF
- Text normalization: Unicode NFC
- Embedding model: `intfloat/multilingual-e5-base`
- Vector database: ChromaDB
- Local language model: Ollama — `qwen2.5:7b`
- Citation metadata: page, section, chunk ID
- Evaluation output: `outputs/test_run.json`
ls -l outputstail -35 README.md