# GCG e nanoGCG com PyRIT (execução 100% local)

Implementação de uma bateria comparativa de ataques adversariais por **sufixo** usando **GCG** e **nanoGCG**, com geração local do ataque em **DeepSeek-R1-Distill-Llama-8B** e avaliação local via **PyRIT** sobre **Meta-Llama-3-8B-Instruct**.

> **Importante:** este repositório deve ser lido com base no **estado atual do código**. Quando houver divergência entre o texto do ERAD e a implementação, **vale o que está implementado nos arquivos `GCG.py` e `nanoGCG.py`**.

## Visão geral

O projeto executa um experimento em **três fases**:

1. **Download / validação dos modelos**
   - Garante que os modelos necessários estejam no cache local do Hugging Face.

2. **Geração de sufixo adversarial**
   - `GCG.py`: usa uma implementação local de **Greedy Coordinate Gradient (GCG)**.
   - `nanoGCG.py`: usa a biblioteca **`nanogcg`**.

3. **Avaliação com PyRIT**
   - Cada sufixo gerado é aplicado ao respectivo prompt da bateria.
   - O modelo alvo responde localmente.
   - O experimento mede sucesso do ataque, tempo, RAM, VRAM, tokens gerados e custo estimado.

## O que está implementado hoje

A implementação atual possui as seguintes características:

- Execução **100% local**
- Sem Ollama
- Sem OpenAI
- Sem API externa para inferência
- Integração com:
  - `transformers`
  - `huggingface_hub`
  - `PyRIT`
  - `torch`
  - `psutil`
- Persistência local com `DuckDBMemory`
- Geração de relatório consolidado em JSON
- Medição de:
  - tempo de execução
  - uso médio e pico de RAM
  - pico de VRAM
  - Attack Success Rate (ASR)
  - custo estimado em GPU
  - tokens gerados pelo modelo alvo

## Modelos usados

### Modelo atacante
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

### Modelo alvo
- `meta-llama/Meta-Llama-3-8B-Instruct`

## Divergências em relação ao ERAD

Embora o ERAD descreva a proposta experimental, o repositório já reflete ajustes de implementação que **devem prevalecer**. Entre os principais pontos:

- O alvo efetivamente configurado no código é **`Meta-Llama-3-8B-Instruct`** em ambos os scripts.
- A avaliação do alvo é feita localmente via `transformers`, encapsulada em uma classe `HFLocalTarget` compatível com o PyRIT.
- O fluxo real inclui correções para ambiente offline, carregamento por snapshot local, uso de `chat_template` e compatibilidade com modelos base/instruct.
- A bateria atualmente usada no código possui **5 prompts**, com **1 variação por prompt**.
- O projeto está estruturado como experimento de linha de comando com fases separadas:
  - `download`
  - `gcg`
  - `pyrit`

## Estrutura dos arquivos principais

```text
.
├── GCG.py
├── nanoGCG.py
├── README.md
└── (arquivos gerados em tempo de execução)
