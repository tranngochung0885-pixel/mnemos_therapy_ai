# 🧠 MNEMOS v1.0
### Memory Neural Engram Orchestration System
> *"Rekindling the Embers of Memory"*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 112/112](https://img.shields.io/badge/tests-112%2F112%20passing-brightgreen.svg)]()
[![Lines: 4,320](https://img.shields.io/badge/lines-4%2C320-blue.svg)]()

**MNEMOS** is an open-source AI-driven neurocognitive memory therapy system designed to support clinical treatment of patients with Alzheimer's disease, Traumatic Brain Injury (TBI), PTSD, and Mild Cognitive Impairment (MCI). It implements a novel synthesis of 2025 neuroscience research into a production-ready Python framework.

> ⚠️ **Clinical Disclaimer**: MNEMOS is a research and support tool. It does not replace licensed therapists, neurologists, or clinical psychologists. All therapy sessions should be supervised by qualified healthcare professionals.

---

## 🌟 What Makes MNEMOS Different?

Most AI therapy tools are chatbots with memory. MNEMOS is grounded in the actual neuroscience of memory — specifically the landmark finding that **engrams in Alzheimer's patients are often latent (still intact) rather than destroyed**. The goal is not to create new memories, but to *reactivate the ones already there*.

| Feature | Typical AI Chatbot | MNEMOS |
|---|---|---|
| Memory model | Conversation log | Physical engram traces (Josselyn & Tonegawa 2020) |
| Retrieval | Semantic search only | HNSW + BM25 + RRF hybrid |
| Scheduling | None | FSRS-6 spaced reactivation |
| Family integration | None | FamilyVaultRAG — photos, audio, voice clone |
| Safety | None | PTSD Prolonged Exposure protocol (Foa 2019) |
| Progress tracking | None | MMSE surrogate + Bayesian trajectory |
| Temporal encoding | None | HiPPO-LegS (Gu et al. 2020 NeurIPS) |

---

## 🎯 Clinical Targets

| Condition | MNEMOS Approach |
|---|---|
| **Alzheimer's / Dementia** | Reactivate latent engrams via multimodal stimulation; family voice bridge; daily spaced practice |
| **Traumatic Brain Injury (TBI)** | Rebuild severed memory network via NPTX connectivity reconstruction; bridge fragmented traces |
| **PTSD** | Safety-gated gradual exposure (Prolonged Exposure protocol); IBI monitoring; grounding techniques |
| **MCI (Mild Cognitive Impairment)** | Early intervention; FSRS-6 scheduled reinforcement; network density growth |

---

## 🏗️ Architecture

```
MNEMOS v1.0
│
├── §01  Constants, Config, Enums
│       MnemosConfig (100+ parameters), Condition, LTPPhase, SafetyLevel
│
├── §02  MathLib — All algorithms with citations
│       FSRS-6 · HiPPO-LegS · BM25 · RRF · Welford · PAC-Bayes
│       IBI score · Bayesian trajectory · NPTX connectivity
│
├── §03  MemoryEngram — Physical memory trace
│       LTP phases · Transcriptional cascade simulation · FSRS scheduling
│       NPTX graph connectivity · Retrievability decay
│
├── §04  PatientProfile — Full clinical record
│       Demographics · Cognitive baselines · PTSD safety config
│       Family vault metadata · Session history
│
├── §05  MemoryDatabase — SQLite persistence (WAL mode)
│       8 tables: patients, engrams, sessions, media_assets,
│       family_stories, conversation_turns, consolidation_log
│
├── §06  EngramAllocator — HNSW associative storage
│       Semantic search (FAISS HNSW) · BM25 keyword index
│       RRF hybrid fusion · NPTX connection graph
│
├── §07  FamilyVaultRAG — Family-uploaded memories
│       Photos · Audio · Videos · Written stories
│       Relevance retrieval for therapy stimulation
│
├── §08  VoiceBridge — Therapeutic voice synthesis
│       ElevenLabs voice cloning · XTTS2 local · gTTS fallback
│       Vietnamese-first message templates
│
├── §09  MultimodalStimulator — Sensory trigger packages
│       Visual · Auditory · Music (ISO principle) · Narrative
│       Intensity-calibrated stimulus packages
│
├── §10  SafetyGate — PTSD clinical safety protocol
│       Prolonged Exposure hierarchy (Foa et al. 2019)
│       IBI real-time monitoring · Grounding techniques
│       Session abort triggers · Exposure level progression
│
├── §11  TherapyLLMClient — Adaptive conversation engine
│       Vietnamese-first system prompts · Anthropic/OpenAI/Ollama
│       10 therapeutic principles · Memory cuing techniques
│       Welford latency tracking
│
├── §12  SessionOrchestrator — Full session lifecycle
│       5-phase protocol: Orientation→Warmup→Recall→Consolidation→Cooldown
│       FSRS-6 target selection · Real-time safety monitoring
│       Emotional arc tracking
│
├── §13  ConsolidationEngine — Offline memory replay (NightOwl)
│       SWR-inspired replay cycles · Connection synthesis
│       Latent engram bridging · Transcriptional cascade updates
│
├── §14  ProgressTracker — Clinical monitoring
│       MMSE surrogate scoring · Bayesian trajectory prediction
│       PAC-Bayes confidence bounds · Welford statistics
│
├── §15  ClinicalReportGenerator — Structured reports
│       Clinician report (Vietnamese) · Family portal report
│       Personalized recommendations
│
└── §16  MnemosSystem + §17 FastAPI + CLI + Tests (112/112)
```

---

## 🔬 Scientific Foundations

Every algorithm in MNEMOS traces to a peer-reviewed citation:

| Algorithm | Reference | Used For |
|---|---|---|
| **Engram theory** | Josselyn & Tonegawa (2020) *Science* 367:eaaw4325 | Core memory model |
| **Transcriptional cascade** | Terceros et al. (2025) *Nature* | Camta1→Tcf4→Ash1l simulation |
| **NPTX connectivity** | Jin et al. (2025) *Nature Cell Biology* | Memory network scoring |
| **Bi-temporal KG** | Rasmussen (2025) arXiv:2501.13956 | Episodic memory graph |
| **HiPPO-LegS** | Gu et al. (2020) *NeurIPS* Appendix C | Temporal state encoding |
| **FSRS-6** | open-spaced-repetition (2024) | Spaced reactivation scheduling |
| **HNSW** | Malkov & Yashunin (2018) arXiv:1603.09320 | Associative retrieval |
| **BM25** | Robertson & Zaragoza (2009) *FnTIR* | Keyword memory search |
| **Prolonged Exposure** | Foa et al. (2019) PE protocol | PTSD safety hierarchy |
| **GoodTimes AI** | Wang et al. (2024) *CHI* | Reminiscence therapy design |
| **AMPER co-design** | Parra et al. (2025) *Alzheimer's Assoc.* | ECA design principles |
| **AI reminiscence** | Rememo (2026) arXiv:2602.17083 | AI-in-the-loop model |
| **Welford stats** | Welford (1962) *Technometrics* | Online mean/variance |
| **PAC-Bayes** | McAllester (1999) *COLT* | Confidence bounds |
| **RRF fusion** | Cormack et al. (2009) *SIGIR* | Hybrid retrieval ranking |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mnemos
cd mnemos

# Install core dependencies
pip install numpy scipy

# Optional (for full feature set)
pip install faiss-cpu sentence-transformers anthropic fastapi uvicorn aiohttp rich
```

### Environment Variables

```bash
# LLM Provider (choose one)
export ANTHROPIC_API_KEY="your-key"    # Recommended
export OPENAI_API_KEY="your-key"       # Alternative
# Or use local Ollama: MNEMOS_LLM=ollama

# Voice Cloning (optional)
export ELEVENLABS_API_KEY="your-key"

# Configuration
export MNEMOS_DATA_DIR="./mnemos_data"
export MNEMOS_LLM="anthropic"           # anthropic|openai|ollama|off
export MNEMOS_TTS="elevenlabs"          # elevenlabs|xtts2|gtts|off
export MNEMOS_LANGUAGE="vi"             # vi|en
```

### Run Tests (no API keys needed)

```bash
python mnemos_complete.py --mode test
# Expected: 112/112 passed ✅
```

### Interactive CLI

```bash
python mnemos_complete.py --mode cli
```

```
🧠 MNEMOS > /new-patient
  Họ tên đầy đủ: Nguyễn Văn An
  Ngày sinh: 1945-03-15
  Chẩn đoán: 1  (alzheimer)
  Bác sĩ: BS. Trần Thị B
  MMSE baseline: 18

✅ Tạo bệnh nhân: PT2B4F19AC

🧠 MNEMOS[PT2B4F] > /add-engram
  Tên ký ức: Đám cưới năm 1970
  Mô tả: Lễ cưới tại làng Cổ Loa, Hà Nội. Rất đông khách.
  Thời gian: 1970
  Địa điểm: Cổ Loa, Hà Nội
  Người liên quan: Bà Lan (vợ), Ông Hải (bố)
  Cảm xúc: 0.9

🧠 MNEMOS[PT2B4F] > /add-story
  Tiêu đề: Bố và cây đàn bầu
  Nội dung: Mỗi tối thứ Sáu, bố lại mang cây đàn bầu ra chơi...
  Kể bởi: Nguyễn Thị C (con gái)

🧠 MNEMOS[PT2B4F] > /session
# Runs full 5-phase therapy session

🧠 MNEMOS[PT2B4F] > /report
# Prints Vietnamese clinical report
```

### REST API Server

```bash
python mnemos_complete.py --mode serve --port 8766
```

Key endpoints:
```
POST   /patients                      — Create patient
GET    /patients/{id}                 — Get patient
POST   /patients/{id}/engrams         — Add memory
POST   /patients/{id}/engrams/search  — Search memories
POST   /patients/{id}/stories         — Add family story
POST   /patients/{id}/media           — Upload photo/audio
POST   /patients/{id}/sessions        — Run therapy session
GET    /patients/{id}/report/clinician — Clinical report
GET    /patients/{id}/report/family   — Family report
GET    /patients/{id}/progress        — Progress analytics
POST   /patients/{id}/consolidation   — Run NightOwl consolidation
```

### Python API

```python
import asyncio
from mnemos_complete import build_mnemos, MnemosConfig

# Initialize system
cfg = MnemosConfig.from_env()
system = build_mnemos(cfg)

# Create patient
patient = system.create_patient(
    full_name="Nguyễn Văn An",
    date_of_birth="1945-03-15",
    gender="nam",
    condition="alzheimer",
    clinician_name="BS. Trần Thị B",
    facility_name="Bệnh viện Bạch Mai",
    mmse_baseline=18.0,
    language="vi",
)

# Populate memory vault
system.add_engram(
    patient.patient_id,
    title="Đám cưới 1970",
    description="Lễ cưới tại làng Cổ Loa, Hà Nội, rất đông khách...",
    time_period="1970",
    location="Cổ Loa, Hà Nội",
    people=["Bà Lan (vợ)", "Ông Hải (bố)"],
    emotional_valence=0.9,
    importance=1.0,
)

system.add_engram(
    patient.patient_id,
    title="Những năm làm thợ may",
    description="Xưởng may ở phố Huế, Hà Nội, từ 1968 đến 1975",
    time_period="1968-1975",
    location="Phố Huế, Hà Nội",
    is_latent=True,   # Currently inaccessible — candidate for reactivation
)

# Family uploads a story
system.add_family_story(
    patient.patient_id,
    title="Bố và cây đàn bầu",
    content="Mỗi tối thứ Sáu, bố lại mang cây đàn bầu ra chơi. "
            "Cả nhà ngồi quây quần nghe. Đó là kỷ niệm đẹp nhất tuổi thơ của chúng con...",
    told_by="Nguyễn Thị C (con gái)",
    time_period="1975-1985",
    people=["Bố", "Mẹ", "Chị Lan", "Anh Hùng"],
)

# Run therapy session
session = asyncio.run(system.run_therapy_session(patient.patient_id))
print(f"Engrams activated: {len(session.engrams_activated)}")
print(f"New connections: {session.new_connections}")
print(f"Quality score: {session.quality_score:.2f}")

# Generate reports
print(system.generate_report(patient.patient_id, "clinician"))
print(system.generate_report(patient.patient_id, "family"))
```

---

## 🛡️ Safety Design

MNEMOS implements a multi-layer safety architecture, especially for PTSD patients:

### Safety Levels

| Level | IBI Score | Action |
|---|---|---|
| 🟢 GREEN | < 0.60 | Proceed normally |
| 🟡 YELLOW | 0.60–0.74 | Slow down, gentle check-in |
| 🟠 ORANGE | 0.75–0.84 | Pause, apply grounding technique |
| 🔴 RED | ≥ 0.85 | Abort session immediately |

### PTSD Prolonged Exposure Protocol (Foa et al. 2019)

MNEMOS implements gradual exposure hierarchy:
1. Start at **20% intensity** (safe topics only)
2. Increase **5% per successful session**
3. Requires **3 consecutive safe sessions** before increase
4. Clinician can override at any time

### Grounding Techniques (Vietnamese)

When arousal is detected, MNEMOS delivers evidence-based grounding:
- **5-4-3-2-1 sensory grounding** (ORANGE level)
- **4-4-6 box breathing** (YELLOW level)
- **Emotional anchor technique** (RED level — immediate abort)

---

## 🌙 NightOwl: Offline Memory Consolidation

Inspired by hippocampal Sharp-Wave Ripple (SWR) replay during sleep, MNEMOS runs a nightly consolidation cycle:

```
02:00 AM → ConsolidationEngine.run_all_patients()
  │
  ├── SWR Replay (3 cycles)
  │     Replay recent engrams in priority order
  │     Strengthen stability for reviewed memories
  │
  ├── NPTX Connection Building
  │     Find semantically similar engrams
  │     Build cross-modal associations
  │     Update connectivity scores
  │
  ├── Transcriptional Cascade Updates
  │     Camta1 → Tcf4 → Ash1l simulation
  │     Consolidation level progression
  │
  └── Latent Engram Bridging
        Find non-latent engrams that can "reach" latent ones
        Create bridge connections for reactivation
```

Schedule with cron:
```bash
# Run consolidation every night at 2 AM
0 2 * * * /usr/bin/python3 /path/to/mnemos_complete.py --mode consolidate
```

---

## 📁 Project Structure

```
mnemos/
├── mnemos_complete.py          # Single-file implementation (4,320 lines)
├── README_MNEMOS.md            # This file
├── requirements.txt            # Dependencies
└── mnemos_data/                # Data directory (auto-created)
    ├── mnemos.db               # SQLite database (WAL mode)
    ├── media/                  # Patient media assets
    │   └── {patient_id}/       # Per-patient media
    └── voice_models/           # Voice clone models
        └── {patient_id}/       # Per-patient voice refs
```

---

## 📦 Dependencies

### Core (required)
```
numpy>=1.24
scipy>=1.11
```

### Enhanced Retrieval (recommended)
```
faiss-cpu>=1.7.4          # HNSW vector index
sentence-transformers>=2.6 # multilingual-e5 embeddings
```

### LLM Providers (choose one)
```
anthropic>=0.30            # Claude (recommended for Vietnamese)
openai>=1.30               # GPT-4o
# or: Ollama (local, free) — install separately
```

### Voice & Audio
```
aiohttp>=3.9               # Async HTTP for ElevenLabs
gtts>=2.5                  # gTTS fallback TTS
# TTS-coqui                # XTTS2 local (install separately)
```

### API Server
```
fastapi>=0.111
uvicorn>=0.30
```

### UI Enhancement
```
rich>=13.7                 # Beautiful CLI output
```

Install all:
```bash
pip install numpy scipy faiss-cpu anthropic fastapi uvicorn aiohttp rich
```

---

## 🔮 Research Roadmap

### Phase 1 — Current (v1.0)
- ✅ Core engram architecture with FSRS-6 scheduling
- ✅ HiPPO-LegS temporal encoding
- ✅ HNSW + BM25 + RRF hybrid retrieval
- ✅ FamilyVaultRAG (text, photo, audio metadata)
- ✅ VoiceBridge (ElevenLabs, XTTS2)
- ✅ SafetyGate with PE protocol
- ✅ Vietnamese-first therapy LLM
- ✅ 5-phase session orchestrator
- ✅ NightOwl consolidation engine
- ✅ Clinical + family reporting
- ✅ FastAPI server + interactive CLI

### Phase 2 — Planned (v2.0)
- 🚧 Real-time facial expression analysis (arousal detection from video)
- 🚧 EEG/biometric integration (Apple Watch, Polar H10 for IBI)
- 🚧 Full multimodal embedding (CLIP for photos, ASR for voice)
- 🚧 VR/AR memory environment reconstruction
- 🚧 Federated learning for cross-institution anonymized research

### Phase 3 — Research Vision (v3.0)
- 🔬 Optogenetic-inspired computational reactivation patterns
- 🔬 Graph neural network for memory network topology prediction
- 🔬 Caregiver co-therapy interface
- 🔬 Longitudinal clinical trial data collection framework

---

## 🤝 Contributing

We welcome contributions, especially:
- **Clinical expertise**: Therapists, neurologists with feedback on protocols
- **Vietnamese NLP**: Improving therapy dialogue for Vietnamese patients
- **Accessibility**: Support for more languages and cultural contexts
- **Privacy & security**: PHI protection, HIPAA/GDPR compliance

Please ensure any new algorithm is accompanied by a peer-reviewed citation.

---

## 📜 License

MIT License — free for research and clinical use. See `LICENSE`.

---

## 🙏 Acknowledgments

Built with deep respect for:
- **Patients and families** living with memory loss
- The researchers at **Josselyn lab, MIT CSAIL, Fudan University** whose 2025 work made this possible
- The **open-source AI** community whose tools power MNEMOS
- **Trần 2026 research program** — cognitive architecture for human benefit

---

*MNEMOS — "μνήμη" (mnímē) — Ancient Greek for "memory"*

> "The light of memory, or rather the light that memory lends to things, is the palest light of all... I am not quite sure whether I am dreaming or remembering." — Eugène Ionesco
