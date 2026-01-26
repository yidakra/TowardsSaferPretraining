---
marp: true
style: |
  section p,
  section li {
    font-size: 28px;
  }
author: Miguel A. Mendes
theme: default
#class:
#  - lead
#  - invert
paginate: true
size: 16:9
_header: ""
header: UvA, FACT Project. Deliev, Bilge and Mendes 
# header-includes: \usepackage{xcolor}
# header-includes: \documentclass{beamer}
_paginate: false



---


<style>
img[alt~="down-right"] {
  position: absolute;
  top: 490px;
  right: 50px;
}
</style>
<style>
img[alt~="up-left"] {
  position: absolute;
  top: 50px;
  right: 930px;
}
</style>

<!--Left hand side -->

![w:400](image-23.png) 

## <span style="color:blue;">Revisiting Web-Scale Harmful Content Filtering for Safer LLM Pretraining</span>



## Ark Deliev, Ali Bilge and Miguel Mendes</span>

### Master in Artificial Intelligence, FACT Course Project
January, 2026

![bg](image-25.png)


---

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>


<style>
section {
  background: white;
}
</style>


# Project
- We evaluate the reproducibility of a taxonomy-driven framework for harmful content detection using released benchmarks and models.
- The project is based on the following paper: 

[1]. Sai Krishna Mendu , Harish Yenala , Aditi Gulati , Shanu Kumar , Parag Agrawal, (2025), *Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale Datasets for Responsible LLMs*, 2025 IJCAI Conference, arXiv:2505.02009v3. 

<hr class="sep">

<style>
  hr.sep {
    border: none;
    border-top: 2px solid #cccccc;
    margin: 24px 0;
  }
</style>



> Abbreviations
>- **TTP** — Topical and Toxic Prompt  
>- **HAVOC** — Multi-Harm Open-ended Toxicity Benchmark  
>- **LLM** — Large Language Models




---
# Why this paper matters?
<br><br>

![w:900 center](Migi_tikz.svg)

---
# Claims made in the original paper
<style>
  /* Slide padding */
  section {
    padding: 60px 70px;
  }

  /* Box styling */
  .box {
    border: 2px solid rgba(0,0,0,0.12);
    border-radius: 14px;
    padding: 18px 22px;
    margin: 14px 0;
    font-size: 32px;
    line-height: 1.25;
    background: rgba(255,255,255,0.92);
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  }

  /* Optional accent per line */
  .accent1 { border-left: 10px solid #2E86AB; }
  .accent2 { border-left: 10px solid #1B998B; }
  .accent3 { border-left: 10px solid #F4A261; }
  .accent4 { border-left: 10px solid #E63946; }

  /* Make the slide title look good */
  h1 {
    font-size: 44px;
    margin-bottom: 20px;
  }
</style>


<div class="box accent1"><b><span style="color:blue;">Claim 1:</span></b> <b>TTP</b> performs well on <b>TTP-Eval</b></div>

<div class="box accent2"><b><span style="color:darkgreen;">Claim 2:</span></b> <b>HarmFormer</b> shows strong performance</div>

<div class="box accent3"><b><span style="color:darkorange;">Claim 3: </span></b><b>TTP</b> outperforms baselines on OpenAIModeration</div>

<div class="box accent4"><b><span style="color:red;">Claim 4: </span></b><b>HAVOC</b> shows ~26.7% leakage</div>

---
# Reproduction Setup

![w:800 center](Migi_Datasets.svg)



---
# Key Result 1: TTP on TTP-Eval

![w:900 center](performance_comparison.svg)

---
# Key Result 2: HarmFormer vs TTP
<br>

![w:900 center](f1_score_hbar.svg)


---
# Key Result 3: HAVOC (Successfully Reproduced)


![w:680 center](havoc_leakage_by_prompt_type.svg)

---
# Extension: Cross-Model TTP Failure 
<br>



| Model | Prompt runs | Output parsable | Infrastructure | Result |
|:------|:-----------:|:---------------:|:--------------:|:------:|
| GPT-4o | ✓ | ✓ | ✓ | ✅ Works |
| Gemini 2.0 | ✓ | ✓ | ⚠️ Budget | ⚠️ Partial |
| Gemma 2 27B | ✓ | ✗ | ✓ | ❌ Format |



---

# Why Reproducibility Failed


---
# Environmental & Practical Impact
<br>

| Aspect | Original paper | Our reproduction |
|:------|:---------------|:------------------|
| **Web pages processed** | ~3,000,000 | 393 |
| **Model training** | Yes (HarmFormer trained) | No (pretrained) |
| **HAVOC inference** | Yes (multiple models) | No (precomputed) |
| **GPU hours** | O(10,000+) (estimate) | ~5–15 |
| **CO₂** | Not reported | ~0.05 kg |



---
# Conclusions and Future Work

- <span style="color:blue;">**Key Conclusions**</span>
    - HAVOC is reliable and easily reproducible
    - HarmFormer generalizes reasonably to human-annotated data
    - TTP performance is model-dependent and fragile

- <span style="color:darkgreen;">**Future work**</span>
    - Future safety tools should prioritize open, robust, and model-agnostic designs.
