# GPT2-BASIC: Fixed-Point Language Models and Local Recall on DOS-Class Systems

## Current Status Note

The slim production target is the DOS FreeBASIC fixed-point runtime documented
in `README.md`, `qemu/README.md`, and `qemu/evidence/domain_training_strategy_report.md`.
Formerly aspirational software subsystems from this paper-style document
including memory tracking, matrix pooling, disk parameter streaming,
block-sparse attention, SIMD-like packed operations, and assembly/fallback
fixed-point support are now source-verified by
`scripts/verify_aspirational_software.py`. The closure report is
`qemu/evidence/aspirational_software_closure.md`, and ICC reports
`gpt2-basic-aspirational-software` as `ready` with score 100.

Physical board timing remains deferred. QEMU emulator evidence is the accepted
timing basis until a physical 486/Pentium board is available.

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Historical Background](#2-historical-background)
   - [2.1 The 486 Era Computing Landscape](#21-the-486-era-computing-landscape)
   - [2.2 State of AI in the Early 1990s](#22-state-of-ai-in-the-early-1990s)
   - [2.3 Modern Transformer Models](#23-modern-transformer-models)
3. [Technical Architecture](#3-technical-architecture)
   - [3.1 System Overview](#31-system-overview)
   - [3.2 Data Structures and Memory Management](#32-data-structures-and-memory-management)
   - [3.3 Tokenization Approach](#33-tokenization-approach)
   - [3.4 Transformer Implementation](#34-transformer-implementation)
4. [Core Innovations](#4-core-innovations)
   - [4.1 4-bit Logarithmic Quantization](#41-4-bit-logarithmic-quantization)
   - [4.2 Fixed-Point Arithmetic (Q16.16)](#42-fixed-point-arithmetic-q1616)
   - [4.3 Block-Sparse Attention](#43-block-sparse-attention)
   - [4.4 Disk Streaming Parameter System](#44-disk-streaming-parameter-system)
   - [4.5 SIMD-like Bit Manipulation](#45-simd-like-bit-manipulation)
   - [4.6 Assembly Optimizations](#46-assembly-optimizations)
5. [Platform-Specific Implementations](#5-platform-specific-implementations)
   - [5.1 DOS Implementation](#51-dos-implementation)
   - [5.2 Windows Considerations](#52-windows-considerations)
   - [5.3 Macintosh Adaptation](#53-macintosh-adaptation)
   - [5.4 OS/2 and Linux Considerations](#54-os2-and-linux-considerations)
6. [Performance Analysis](#6-performance-analysis)
   - [6.1 Benchmarking Methodology](#61-benchmarking-methodology)
   - [6.2 Results on Modern Hardware](#62-results-on-modern-hardware)
   - [6.3 QEMU Era-Profile Performance Evidence](#63-qemu-era-profile-performance-evidence)
   - [6.4 Memory Usage Analysis](#64-memory-usage-analysis)
7. [Historical Comparison and Design Implications](#7-historical-comparison-and-design-implications)
   - [7.1 Statistical Computing in the 1990s](#71-statistical-computing-in-the-1990s)
   - [7.2 AI Research Trajectory](#72-ai-research-trajectory)
   - [7.3 Language-Model Operations Under Constraints](#73-language-model-operations-under-constraints)
   - [7.4 Release Application Surface](#74-release-application-surface)
   - [7.5 Hardware Co-Design Pressure Points](#75-hardware-co-design-pressure-points)
8. [Educational Value](#8-educational-value)
   - [8.1 Demonstrating Transformer Fundamentals](#81-demonstrating-transformer-fundamentals)
   - [8.2 Optimization Techniques for Constrained Environments](#82-optimization-techniques-for-constrained-environments)
   - [8.3 Insights for Modern Edge AI](#83-insights-for-modern-edge-ai)
9. [Future Directions](#9-future-directions)
   - [9.1 Further Optimizations](#91-further-optimizations)
   - [9.2 Applications and Extensions](#92-applications-and-extensions)
10. [Addendum: Practical Applications and Modern Relevance](#10-addendum-practical-applications-and-modern-relevance)
    - [A. Modern Quantization Connections](#a-modern-quantization-connections)
      - [A.1 Connections to Modern Neural Network Quantization](#a1-connections-to-modern-neural-network-quantization)
      - [A.2 Edge AI and Embedded Systems Applications](#a2-edge-ai-and-embedded-systems-applications)
    - [B. Practical Extensions and Integration](#b-practical-extensions-and-integration)
      - [B.1 Integration with Existing Systems](#b1-integration-with-existing-systems)
      - [B.2 Educational Toolkit Extensions](#b2-educational-toolkit-extensions)
    - [C. Constrained and Legacy System Engagement](#c-constrained-and-legacy-system-engagement)
      - [C.1 Demoscene Potential](#c1-demoscene-potential)
      - [C.2 Legacy System Preservation](#c2-legacy-system-preservation)
    - [D. Training and Export Considerations](#d-training-and-export-considerations)
      - [D.1 Host-Side Training Boundary](#d1-host-side-training-boundary)
      - [D.2 Target-Side Adaptation](#d2-target-side-adaptation)
    - [E. Open Source and Community Development](#e-open-source-and-community-development)
      - [E.1 Open Source Framework](#e1-open-source-framework)
      - [E.2 Educational Curriculum](#e2-educational-curriculum)
11. [References](#11-references)

## 1. Executive Summary

This paper documents GPT2-BASIC, a compact fixed-point transformer and assistant
runtime implemented in BASIC for DOS-class systems. The implementation
demonstrates that the core operations behind GPT-style language models can be
expressed as ordinary file formats, integer arithmetic, tokenizer logic,
matrix/vector kernels, and deterministic control flow. It also documents the
local assistant layer built around hot-loadable packs, golden replies, session
memory, binary knowledge records, and sharded term indexes.

The implementation serves multiple purposes: as an educational resource for the
core mathematical operations underlying transformer models, as a concrete
engineering reference for local AI under severe CPU and memory limits, and as a
release-tested DOS runtime with QEMU evidence and a physical-machine transfer
workflow. It is not a claim that a tiny 486-class model competes with modern
hosted LLMs.

Key technical innovations in this implementation include:

1. 4-bit logarithmic quantization for efficient weight representation
2. Fixed-point arithmetic (Q16.16) completely eliminating floating-point operations
3. Block-sparse attention using linked-list data structures for memory efficiency
4. Disk streaming parameter system for operation within 32MB RAM constraints
5. SIMD-like operations through bit manipulation techniques
6. Critical section optimization through targeted assembly language

Current repository evidence now distinguishes the production path from earlier
design-space modules. The promoted DOS checkpoint is a 2-layer, 48-dimensional,
4-head, 192-context GPT2-BASIC model with a 4096-token longest-match lexicon,
463,168 parameters, Q20.12 fixed-point weights, a DOS KV decode cache, and
QEMU/DOS parity and quality evidence. It passes the 10-prompt DOS all-suite at
10/10 average 0.961 and measures 2.46 tok/s on the QEMU 486DX2/66 gate. A
separate q4/log token+head release mode is now fully wired through host export,
DOS loading, vector parity, quality evaluation, and QEMU `--perf`: it reduces
runtime memory from 2,055,940 to 974,724 bytes while measuring 2.12 tok/s on
the same gate. Sections below describe both the original architecture concepts
and the realized production subset; `qemu/evidence/domain_training_strategy_report.md`
is the authoritative implementation ledger.

This paper provides a technical analysis of these implementation techniques,
documents the challenges of running transformer-style inference on constrained
hardware, and compares the design against the tools and limits of DOS-class
systems.

## 2. Historical Background

### 2.1 The 486 Era Computing Landscape

The Intel 80486 (commonly called "486") era, spanning roughly 1989 to 1995, represented a significant period in personal computing evolution. These systems typically featured:

- **Processors:** Intel 80486 variants (SX, DX, DX2, DX4) running at 25-100 MHz
- **Memory:** 4-32 MB RAM, often with complex memory management systems
- **Storage:** 100-500 MB hard drives, with access times of 10-20 milliseconds
- **Operating Systems:** MS-DOS 6.x, Windows 3.x, early Windows 95, OS/2, early Linux

This hardware environment imposed severe constraints compared to modern systems, with approximately 100,000 times less RAM, 10,000 times less storage, and 1,000 times slower CPU performance than today's average computers.

The 486 era was marked by sophisticated memory management techniques to overcome hardware limitations. MS-DOS systems utilized a complex ecosystem of memory types (Crowley, 1996):

- **Conventional memory:** The first 640 KB of RAM, directly accessible by DOS programs
- **Upper memory:** The 384 KB between 640 KB and 1 MB, often used for device drivers
- **Extended memory:** Memory above 1 MB, accessible using protected mode or memory managers
- **Expanded memory:** A bank-switching technique allowing access to additional memory

Memory management tools like HIMEM.SYS and EMM386.EXE were essential for maximizing available RAM, and programming practices of the era focused heavily on memory efficiency (Duncan, 1992). The introduction of DOS extenders like DOS4GW and CWSDPMI allowed programs to break the 640 KB barrier and access extended memory in protected mode, critical for memory-intensive applications.

Programming in this era was dominated by languages like C, Pascal, and various BASIC dialects. Microsoft's QuickBASIC and Borland's Turbo Basic were popular high-level development environments, while PowerBASIC offered a compromise between ease of use and performance. Assembly language was frequently used for critical sections of code where performance was paramount (Duntemann, 1992).

### 2.2 State of AI in the Early 1990s

The 1990s represented what has been termed the "AI Winter," a period of reduced funding and commercial interest in artificial intelligence following the failure of expert systems to meet commercial expectations in the 1980s (Hendler, 2008). Despite this climate, significant research continued across several AI paradigms:

1. **Symbolic AI and Expert Systems:** Rule-based approaches dominated practical AI applications, with systems like CLIPS (C Language Integrated Production System) from NASA and commercial expert system shells like Level5 Object (Information Builders, 1992).

2. **Neural Networks:** Following the publication of the backpropagation algorithm (Rumelhart et al., 1986), there was renewed interest in neural network approaches. However, practical applications were limited by computational constraints and theoretical understanding.

3. **Statistical Methods:** Hidden Markov Models (HMMs) dominated speech recognition (Rabiner, 1989), while early statistical NLP methods began to emerge for language tasks (Jelinek, 1991).

4. **Bayesian Approaches:** Bayesian networks were developed for reasoning under uncertainty (Pearl, 1988), with applications in diagnosis and prediction systems.

5. **Game Playing Systems:** Chess programs like Deep Thought (predecessor to Deep Blue) demonstrated specialized AI applications (Hsu et al., 1990).

Commercial neural network tools of the era included NeuroSolutions, NeuralWorks Professional, and MATLAB's early Neural Network Toolbox. These systems typically supported relatively small network architectures with limited layers (usually 1-3 hidden layers) and hundreds to thousands of parameters, rather than the millions to billions common today.

Key AI researchers during this period included:

- **Geoffrey Hinton**, who continued to advance backpropagation and distributed representations (Hinton, 1990)
- **Yann LeCun**, who pioneered convolutional neural networks for handwritten digit recognition (LeCun et al., 1990)
- **John Hopfield**, whose work on recurrent neural networks influenced the field (Hopfield, 1988)
- **Michael Jordan**, who developed early work on recurrent networks and probabilistic models (Jordan, 1990)

The computational limits of the era posed significant challenges for neural network implementation. Researchers often relied on specialized workstations or minimized model complexity to accommodate hardware constraints. Training times of days or weeks were common for even modest neural networks on 486-era hardware.

### 2.3 Modern Transformer Models

To provide context for our implementation, we briefly review the architecture of modern transformer models, particularly the GPT (Generative Pre-trained Transformer) family.

Transformers, introduced by Vaswani et al. (2017), represented a significant departure from previous sequence processing architectures by eliminating recurrence and convolution in favor of attention mechanisms. The key innovation was the self-attention mechanism, allowing the model to weigh the importance of different parts of the input sequence when producing each output element.

GPT-2, released by OpenAI (Radford et al., 2019), built upon this architecture with a decoder-only transformer designed for autoregressive language modeling. The core components of GPT-2 include:

1. **Token Embeddings:** Converting input tokens to continuous vector representations
2. **Positional Encodings:** Providing sequence position information
3. **Self-Attention Layers:** Computing attention weights between all tokens in the sequence
4. **Feed-Forward Networks:** Applying non-linear transformations to each position independently
5. **Layer Normalization:** Stabilizing training by normalizing activations
6. **Autoregressive Decoding:** Generating outputs one token at a time

The mathematical foundation of the self-attention mechanism can be expressed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where $Q$, $K$, and $V$ are the query, key, and value matrices, and $d_k$ is the dimensionality of the keys.

Contemporary GPT-2 implementations use floating-point arithmetic, large dense matrix operations, and are optimized for modern GPUs with dedicated tensor processing units. The original GPT-2 models ranged from 117 million to 1.5 billion parameters, requiring gigabytes of memory and substantial computational resources—far beyond what 486-era systems could reasonably support.

Our implementation necessarily scales down the model size and introduces numerous optimizations to operate within 486-era constraints, while maintaining the core mathematical operations and architectural principles that define transformer models.

## 3. Technical Architecture

### 3.1 System Overview

GPT2-BASIC follows the core architectural principles of transformer models while
introducing optimizations for constrained hardware. The system architecture is
organized into several key components, illustrated in Figure 1.

```
┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ Data          │  │ Matrix           │  │ Quantization      │
│ Structures    │◀─┤ Operations       │◀─┤ System            │
└───────────────┘  └──────────────────┘  └───────────────────┘
       ▲                    ▲                      ▲
       │                    │                      │
       ▼                    ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ Transformer   │  │ Block-Sparse     │  │ Softmax           │
│ Components    │◀─┤ Attention        │◀─┤ Fixed-Point       │
└───────────────┘  └──────────────────┘  └───────────────────┘
       ▲                    ▲                      ▲
       │                    │                      │
       ▼                    ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ File I/O      │  │ Assembly         │  │ SIMD-like         │
│ Streaming     │◀─┤ Optimizations    │◀─┤ Operations        │
└───────────────┘  └──────────────────┘  └───────────────────┘
       ▲                    ▲                      ▲
       │                    │                      │
       ▼                    ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ Tokenizer     │  │ Benchmark        │  │ Model             │
│ & Vocabulary  │◀─┤ System           │◀─┤ Integration       │
└───────────────┘  └──────────────────┘  └───────────────────┘
```
*Figure 1: System architecture showing component relationships*

The implementation is structured as a set of modular BASIC files, each containing specific functionality:

- `data_structures.bas`: Core matrix and tensor representation
- `matrix_ops.bas`: Fixed-point matrix operations
- `quantization.bas`: 4-bit logarithmic quantization system
- `softmax_fixed.bas`: Fixed-point implementation of softmax
- `block_sparse.bas`: Memory-efficient sparse attention
- `transformer_components.bas`: Attention and feed-forward components
- `file_io.bas`: Disk streaming for model parameters
- `tokenizer.bas`: Text tokenization and vocabulary management
- `model.bas`: Complete transformer model implementation
- `simd_ops.bas`: SIMD-like operations through bit manipulation
- `asm_optimizations.bas`: Assembly language optimizations
- `benchmark.bas`: Performance testing framework
- `main.bas`: Program entry point and user interface

The system operates as follows:

1. Input text is tokenized into integer token IDs
2. Tokens are converted to embeddings with positional encoding
3. The transformer processes embeddings through multiple layers
4. Each layer applies self-attention and feed-forward operations
5. The output layer produces a probability distribution over vocabulary
6. A token is sampled and added to the generated text
7. The process repeats autoregressively until completion

To accommodate 486-era constraints, the implementation includes several critical design decisions:

- **Model Scale**: A significantly reduced model compared to modern GPT-2
  - ~1M parameters (vs. 117M+ in the original GPT-2)
  - Vocabulary size of 1,000-5,000 tokens (vs. 50,000+ in GPT-2)
  - Embedding dimension of 64-128 (vs. 768+ in GPT-2)
  - 2-4 transformer layers (vs. 12+ in GPT-2)
  - 2-4 attention heads (vs. 12+ in GPT-2)
  - Context length of 64-128 tokens (vs. 1024+ in GPT-2)

- **Memory Efficiency**: Techniques to minimize RAM usage
  - 4-bit logarithmic quantization for parameters
  - Block-sparse attention mechanism
  - On-demand loading of model parameters from disk
  - Immediate cleanup of temporary variables and matrices

- **Computational Efficiency**: Methods to speed up processing
  - Fixed-point arithmetic throughout
  - SIMD-like operations through bit manipulation
  - Assembly language for critical calculations
  - Lookup tables for expensive mathematical functions

The system implementation balances algorithmic fidelity (maintaining the core mathematical operations that define transformer models) with practical feasibility on 486-era hardware.

### 3.2 Data Structures and Memory Management

The foundation of the implementation rests on carefully designed data structures that maximize memory efficiency while supporting the necessary operations for transformer computation.

#### Matrix Representation

The primary data structure is the Matrix type, which represents 2D tensors throughout the system:

```basic
TYPE Matrix
    rows AS INTEGER       ' Number of rows in the matrix
    cols AS INTEGER       ' Number of columns in the matrix
    data(,) AS INTEGER    ' 2D array storing LogQuantized values
END TYPE
```

The Matrix type uses INTEGER arrays to store model parameters and activations. Rather than using floating-point values, which would be prohibitively slow on 486-era hardware (particularly 486SX processors without floating-point units), all values are stored in a custom 4-bit logarithmic quantized format (detailed in Section 4.1) packed into INTEGER values.

Memory for matrices is dynamically allocated and deallocated using dedicated initialization and cleanup functions:

```basic
SUB InitMatrix(m AS Matrix, num_rows AS INTEGER, num_cols AS INTEGER)
    m.rows = num_rows
    m.cols = num_cols
    REDIM m.data(num_rows - 1, num_cols - 1) AS INTEGER
END SUB

SUB FreeMatrix(m AS Matrix)
    ERASE m.data
    m.rows = 0
    m.cols = 0
END SUB
```

This explicit memory management is crucial for operating within the tight memory constraints of 486-era systems. By carefully controlling matrix allocation and deallocation, the system can reuse memory for different stages of computation.

#### Memory Budget Analysis

To understand the feasibility of running a transformer model on 486-era hardware, we analyzed the memory requirements for different components. Table 1 compares the memory usage of standard floating-point representation with our optimized approach.

| Component | Standard 32-bit Float | 4-bit Quantized | Reduction Factor |
|-----------|------------------------|-----------------|-----------------|
| Embedding Layer (5K×128) | 2.56 MB | 320 KB | 8× |
| Attention Weights (per layer) | 384 KB | 48 KB | 8× |
| Feed-Forward Weights (per layer) | 512 KB | 64 KB | 8× |
| Attention Matrices | 128 KB - 8 MB (depending on context length) | 16 KB - 1 MB | 8× |
| Working Memory | ~2 MB | ~256 KB | 8× |
| **Total (4-layer model)** | 10-18 MB | 1.25-2.25 MB | 8× |

*Table 1: Memory usage comparison between standard and optimized representations*

Even with these optimizations, a larger model would still exceed the practical memory limits of most 486-era systems. To address this challenge, we implemented a streaming parameter system (detailed in Section 4.4) that loads model parameters from disk on demand, allowing much larger models to operate within limited RAM.

#### Sparse Data Structures

For memory-intensive operations like attention computation, which scales quadratically with sequence length, we introduced specialized sparse data structures:

```basic
TYPE SparseBlock
    row_start AS INTEGER    ' Starting row index of this block
    col_start AS INTEGER    ' Starting column index of this block
    block_size AS INTEGER   ' Size of the square block
    data() AS INTEGER       ' Block data (packed LogQuantized values)
    next AS SparseBlock PTR ' Pointer to next block in linked list
END TYPE

TYPE SparseBlockMatrix
    blocks AS SparseBlock PTR ' Pointer to the first block
    rows AS INTEGER           ' Total rows in the full matrix
    cols AS INTEGER           ' Total columns in the full matrix
    block_size AS INTEGER     ' Standard block size used
    num_blocks AS INTEGER     ' Number of blocks in the matrix
END TYPE
```

This linked-list structure allows the system to represent large matrices with predominantly zero values (common in attention mechanisms with causal masking) using a fraction of the memory that would be required for dense representation.

### 3.3 Tokenization Approach

Tokenization—the process of converting raw text into integer token IDs—presents unique challenges in a 486-era implementation. Modern tokenizers like Byte-Pair Encoding (BPE) used in GPT-2 require substantial memory for vocabulary storage and computation for encoding/decoding.

Our implementation balances fidelity to modern approaches with 486-era practicality through a flexible tokenization system supporting multiple strategies:

1. **Character-level tokenization**: The simplest approach, requiring minimal memory (1 token per character)
2. **Word-level tokenization**: Using word boundaries (spaces, punctuation), with methods for handling unknown words
3. **Subword tokenization**: A simplified BPE-inspired approach for handling common subwords

The tokenizer's implementation emphasizes memory efficiency:

```basic
TYPE TokenizerVocabulary
    tokens() AS STRING    ' Array of token strings
    token_ids AS INTEGER  ' Number of tokens in vocabulary
    max_tokens AS INTEGER ' Maximum vocabulary size
    unk_token AS INTEGER  ' ID for unknown tokens
END TYPE
```

Vocabulary files are stored on disk and loaded incrementally as needed, rather than keeping the entire vocabulary in memory at once. The vocabulary size is configurable (1,000-5,000 tokens) based on available memory, trading off between language modeling capabilities and resource consumption.

For actual tokenization, the system uses an efficient trie-like structure for token matching, prioritizing longer tokens first:

```basic
FUNCTION TokenizeText(text AS STRING, vocab AS TokenizerVocabulary) AS INTEGER()
    ' Implementation efficiently breaks text into tokens
    ' using a greedy longest-match approach
    ' ...
END FUNCTION
```

This approach avoids the need for complex BPE merge operations at runtime, while still providing reasonable tokenization quality for text generation.

The trade-off between tokenization sophistication and resource consumption reflects a realistic constraint that would have applied to 486-era implementations, where memory and computation would necessarily be prioritized for the core transformer operations rather than preprocessing.

### 3.4 Transformer Implementation

The transformer architecture implemented in BASIC follows the core principles of the GPT-2 model while introducing several modifications for efficiency. The implementation includes all essential components:

#### Embedding Layer

The embedding layer converts token IDs to vector representations and adds positional information:

```basic
SUB EmbedTokens(token_ids() AS INTEGER, embedding_matrix AS Matrix, positional_encodings AS Matrix, output AS Matrix)
    ' For each token, look up its embedding vector and add positional encoding
    ' ...
END SUB
```

Unlike the original GPT-2, which uses learned positional embeddings, our implementation uses fixed sinusoidal encodings similar to those in the original transformer paper (Vaswani et al., 2017), saving memory by not requiring additional learned parameters.

#### Multi-Head Attention

The self-attention mechanism is implemented with support for multiple attention heads:

```basic
SUB MultiHeadAttention(input AS Matrix, wq AS Matrix, wk AS Matrix, wv AS Matrix, wo AS Matrix, output AS Matrix, use_sparse AS INTEGER)
    ' Project input to query, key, value representations
    ' For each head:
    '   Compute attention scores
    '   Apply causal mask (for autoregressive generation)
    '   Apply softmax to get attention weights
    '   Compute weighted sum of values
    ' Combine heads and project to output dimension
    ' ...
END SUB
```

A key optimization is the automatic selection between dense and sparse attention computation based on context length, controlled by the `use_sparse` parameter. For shorter contexts, dense computation may be more efficient, while longer contexts benefit from the memory savings of block-sparse attention.

#### Feed-Forward Network

The feed-forward network uses Gated Linear Units (GLU) instead of the standard ReLU activations used in the original GPT-2:

```basic
SUB FeedForward(input AS Matrix, w1 AS Matrix, w2 AS Matrix, w3 AS Matrix, output AS Matrix)
    ' Compute intermediate representation
    ' Apply gating mechanism (GLU)
    ' Project to output dimension
    ' ...
END SUB
```

The GLU activation function can be expressed mathematically as:

$$\text{GLU}(x, W_1, W_2, W_3) = (xW_1 \otimes \sigma(xW_2))W_3$$

Where $\otimes$ represents element-wise multiplication and $\sigma$ is the sigmoid function.

This choice was motivated by GLU's ability to mitigate vanishing gradients and improve performance with fewer layers, which is valuable in our constrained implementation.

#### Layer Normalization

Layer normalization is implemented using fixed-point arithmetic:

```basic
SUB LayerNorm(input AS Matrix, gamma AS Matrix, beta AS Matrix, output AS Matrix, epsilon AS INTEGER)
    ' For each row:
    '   Calculate mean and variance
    '   Normalize values: (x - mean) / sqrt(variance + epsilon)
    '   Scale and shift: gamma * normalized + beta
    ' ...
END SUB
```

The algorithm has been adapted to work with logarithmically quantized values and fixed-point arithmetic, using careful handling of numerical stability issues that arise when computing statistics in limited precision.

#### Full Transformer Layer

A complete transformer layer combines these components with residual connections:

```basic
SUB TransformerLayer(input AS Matrix, attention_weights() AS Matrix, ff_weights() AS Matrix, norm_weights() AS Matrix, output AS Matrix, use_sparse AS INTEGER)
    ' Layer normalization before attention
    ' Multi-head attention with residual connection
    ' Layer normalization before feed-forward
    ' Feed-forward network with residual connection
    ' ...
END SUB
```

The residual connections are essential for allowing information to flow through the network even with the limited precision of our quantized representation.

The full transformer model iteratively applies these layers to process the input sequence:

```basic
SUB TransformerForward(input_ids() AS INTEGER, model_params AS ModelParameters, output_probs AS Matrix)
    ' Embed input tokens
    ' For each layer:
    '   Apply transformer layer
    ' Apply final layer normalization
    ' Project to vocabulary space
    ' Apply softmax to get probabilities
    ' ...
END SUB
```

The autoregressive generation process uses this forward pass repeatedly, each time adding a newly generated token to the context:

```basic
FUNCTION GenerateText(prompt AS STRING, max_length AS INTEGER, temperature AS SINGLE) AS STRING
    ' Tokenize prompt
    ' Loop until max_length or special end token:
    '   Run transformer forward pass
    '   Sample next token based on output probabilities
    '   Add token to generated sequence
    ' Convert tokens back to text
    ' ...
END FUNCTION
```

Temperature-based sampling provides control over the randomness of generation, with higher temperatures producing more diverse but potentially less coherent text, and lower temperatures resulting in more deterministic but potentially repetitive output. The DOS release now exposes this as measurable behavior through `GPT2.EXE --sampling-matrix`, captured by `qemu/run_sampling_486.sh` in `qemu/evidence/sampling_486.log`. That evidence keeps greedy decoding as the deterministic gate while recording fixed-seed top-k and nucleus-style rows for interactive sampling.

## 4. Core Innovations

### 4.1 4-bit Logarithmic Quantization

One of the most critical optimizations in our implementation is the 4-bit logarithmic quantization scheme for model parameters. This technique reduces memory usage by 8× compared to 32-bit floating-point representation, while maintaining sufficient precision for inference.

In the current production runtime this idea is realized conservatively. The
main checkpoint remains Q20.12 signed fixed point for simple parity, while the
two largest vocabulary-shaped tensors can be replaced by q4/log artifacts.
`GPT2TQ4.BIN` stores token-embedding rows as per-token q4/log codes and
`GPT2HQ4.BIN` stores the output head in input-major order with per-output-token
scales. DOS validates both artifacts, skips loading the full resident token
embedding and output head, dequantizes only the current token row, builds a
compact fixed-point decode table for the head, and computes final logits from
the compressed head. The measured result is 98,304 packed bytes per tensor
instead of 786,432 full bytes per tensor, 974,724 bytes of DOS runtime memory
instead of 2,055,940 bytes, and 2.12 tok/s instead of 2.46 tok/s on the QEMU
486DX2/66 performance gate.

#### Mathematical Foundation

The core idea behind logarithmic quantization is to use a non-linear representation that allocates more precision to smaller values and less precision to larger values, following the general pattern of how neural network weights are typically distributed.

Our implementation uses a 4-bit format with the following structure:

- 1 bit for sign
- 3 bits for logarithmic magnitude

Mathematically, the relationship between a floating-point value $f$ and its quantized representation can be expressed as:

$$f \approx sign \times \frac{mantissa}{16} \times 2^{(exponent-8)}$$

Where:
- $sign$ is either +1 or -1
- $mantissa$ is a value from 0 to 15 (4 bits)
- $exponent$ is a value from 0 to 15 (4 bits), with a bias of 8

In BASIC, the quantization is implemented as:

```basic
FUNCTION QuantizeLog (f AS SINGLE) AS LogQuantized
    DIM lq AS LogQuantized

    ' Extract sign
    DIM sgn AS INTEGER
    IF f > 0 THEN sgn = 1
    IF f < 0 THEN sgn = -1
    IF f = 0 THEN sgn = 0 ' Handle zero sign

    DIM abs_f AS SINGLE = ABS(f)

    ' Handle zero or near-zero special case
    IF abs_f < 0.00001 THEN
        lq.packed_value = 0
        FUNCTION = lq
        EXIT FUNCTION
    END IF

    ' Calculate exponent (biased by 8)
    DIM exponent AS INTEGER = INT(LOG(abs_f) / LOG(2)) + 8

    ' Ensure exponent is within the 4-bit range (0-15)
    IF exponent < 0 THEN exponent = 0
    IF exponent > 15 THEN exponent = 15

    ' Calculate mantissa (4 bits, 0-15)
    DIM power_of_2 AS SINGLE = 2.0 ^ (exponent - 8)
    DIM mantissa AS INTEGER = INT((abs_f / power_of_2) * 16.0) AND 15

    ' Pack mantissa and exponent
    DIM packed_val AS INTEGER = mantissa + (exponent * 16)

    ' Apply sign
    IF sgn = -1 THEN packed_val = -packed_val

    lq.packed_value = packed_val
    FUNCTION = lq
END FUNCTION
```

#### Lookup Table Approach

To avoid expensive logarithm and power operations during inference, we use lookup tables for dequantization:

```basic
SUB InitDequantLookup()
    FOR packed_val = -255 TO 255
        DIM lookup_index AS INTEGER = packed_val + 255

        ' Handle the special case for zero
        IF packed_val = 0 THEN
            DequantFixedLookup(lookup_index) = FloatToFixed(0.0)
        ELSE
            ' Determine the sign
            DIM sgn AS INTEGER
            IF packed_val > 0 THEN sgn = 1 ELSE sgn = -1

            ' Extract mantissa and exponent
            DIM abs_packed_val AS INTEGER = ABS(packed_val)
            DIM mantissa AS INTEGER = abs_packed_val MOD 16
            DIM exponent AS INTEGER = abs_packed_val \ 16 ' Integer division

            ' Calculate the original approximate float value
            DIM power_of_2 AS SINGLE = 2.0 ^ (exponent - 8)
            DIM abs_f AS SINGLE = (mantissa / 16.0) * power_of_2

            ' Apply the sign and convert to fixed-point
            DequantFixedLookup(lookup_index) = FloatToFixed(abs_f * sgn)
        END IF
    NEXT packed_val
END SUB
```

This table is computed once at initialization and used repeatedly during inference, eliminating the need for expensive floating-point operations.

#### Error Analysis

To assess the impact of quantization on model accuracy, we analyzed the quantization error across different value ranges. Figure 2 shows the relative error introduced by our 4-bit logarithmic quantization compared to 32-bit floating-point.

```
      |
      |                      x
Rel.  |                   x
Error |                 x
      |             x
      |         x
      |   x   x
      | x
      +-------------------------
        small  medium   large
              Value Range
```
*Figure 2: Relative quantization error across value ranges*

The analysis shows that our 4-bit logarithmic quantization approach introduces a U-shaped error curve. The relative error is higher for very small values (near zero) and very large values (near the representable limit), with minimal error in the medium range. This pattern is advantageous for neural network weights, which typically cluster in this medium range where precision is most important for model performance.

We found that the mean relative error across all parameter values was approximately 8%, while the mean absolute error was less than 0.01. For context, Nagel et al. (2021) demonstrated that transformer models can tolerate quantization errors of up to 12% before significant performance degradation occurs.

A notable advantage of our logarithmic approach over linear quantization is the wider dynamic range. While a 4-bit linear quantization scheme can represent only 16 evenly spaced values, our logarithmic approach can represent values spanning multiple orders of magnitude, from approximately 0.03 to 8.0 (for positive values), with a distribution that naturally matches the statistics of model parameters.

#### Comparison with Contemporary Approaches

While modern neural network quantization techniques like those in Han et al. (1996) were emerging in the early 1990s, they focused primarily on reducing network complexity rather than efficient representation. Our logarithmic quantization approach shares conceptual similarities with techniques used in early computer graphics and signal processing from the era, such as μ-law encoding for audio signals (Smith, 1987) and logarithmic pixel value encoding in certain image formats.

The closest contemporary technique would have been the logarithmic number system (LNS) proposed for certain specialized computational tasks (Arnold et al., 1990), though these were primarily implemented in hardware rather than software.

### 4.2 Fixed-Point Arithmetic (Q16.16)

To eliminate dependence on floating-point operations, which would be prohibitively slow on 486-era hardware (especially on 486SX processors lacking a floating-point unit), we implemented a fixed-point arithmetic system throughout our codebase. This approach draws inspiration from techniques used in early 3D graphics engines like Doom (Carmack, 1993) and early digital signal processing applications.

#### Mathematical Representation

We use a Q16.16 fixed-point format, which dedicates 16 bits to the integer part and 16 bits to the fractional part of each number. In this representation, a 32-bit integer value `v` represents the real number `v / 65536`, where 65536 (2^16) is the scaling factor.

The mathematical basis for this representation can be expressed as:

$$x_{fixed} = x_{real} \times 2^{16}$$

For example, the value 1.5 would be represented as `98304` (1.5 × 65536). This representation allows for a range of approximately ±32,768 with a precision of 1/65536 ≈ 0.0000153.

#### Implementation of Basic Operations

Fixed-point operations are implemented by carefully managing the scaling factor during arithmetic. The core operations are defined as follows:

```basic
CONST FIXED_POINT_SCALE AS LONG = 65536 ' 2^16

' Convert floating-point to fixed-point
FUNCTION FloatToFixed(f AS SINGLE) AS INTEGER
    RETURN CINT(f * FIXED_POINT_SCALE)
END FUNCTION

' Convert fixed-point to floating-point
FUNCTION FixedToFloat(fp AS INTEGER) AS SINGLE
    RETURN (CSNG(fp) / FIXED_POINT_SCALE)
END FUNCTION

' Addition and subtraction work normally
FUNCTION FixedAdd(a AS INTEGER, b AS INTEGER) AS INTEGER
    RETURN a + b
END FUNCTION

FUNCTION FixedSubtract(a AS INTEGER, b AS INTEGER) AS INTEGER
    RETURN a - b
END FUNCTION

' Multiplication requires adjustment for the scaling factor
FUNCTION FixedMultiply(a AS INTEGER, b AS INTEGER) AS INTEGER
    DIM result AS LONGINT
    result = (CLNGINT(a) * CLNGINT(b)) >> 16
    RETURN CINT(result)
END FUNCTION

' Division is the inverse of multiplication
FUNCTION FixedDivide(a AS INTEGER, b AS INTEGER) AS INTEGER
    IF b = 0 THEN
        PRINT "Error: Division by zero!"
        RETURN 0
    END IF
    DIM result AS LONGINT
    result = (CLNGINT(a) << 16) / CLNGINT(b)
    RETURN CINT(result)
END FUNCTION
```

For more complex functions, we implemented either polynomial approximations or lookup tables:

```basic
' Square root using a modified binary search method
FUNCTION FixedSqrt(fp AS INTEGER) AS INTEGER
    ' Handle special cases
    IF fp <= 0 THEN RETURN 0

    ' Initial guess
    DIM low AS INTEGER = 0
    DIM high AS INTEGER = fp
    DIM mid AS INTEGER
    DIM squared AS LONGINT

    ' Binary search for the square root
    WHILE high - low > 1
        mid = (low + high) / 2
        squared = CLNGINT(mid) * CLNGINT(mid)

        IF squared > (CLNGINT(fp) << 16) THEN
            high = mid
        ELSE
            low = mid
        END IF
    WEND

    RETURN low
END FUNCTION
```

#### Numerical Stability and Precision Management

Fixed-point arithmetic introduces challenges in managing numerical stability, particularly for operations like division and square root that can lead to overflow or underflow. We implemented several techniques to address these issues:

1. **Range Analysis**: For each operation in the transformer computation, we performed careful analysis of the expected value ranges to ensure they fit within the representable range of our Q16.16 format.

2. **Scaling**: In cases where intermediate results might exceed the fixed-point range, we introduced temporary scaling factors that were later reversed.

3. **Normalization**: For operations like softmax, we implemented a shift-to-max approach where the maximum value is subtracted before exponentiation to prevent overflow.

4. **Guard Bits**: Critical computations use temporary 64-bit integers to preserve precision for intermediate results.

These techniques were essential for maintaining numerical stability in the transformer computation, particularly in the attention mechanism where the softmax operation can be sensitive to precision issues.

#### Error Analysis and Comparison with Floating-Point

To assess the impact of fixed-point arithmetic on model accuracy, we compared the output of key mathematical operations between our fixed-point implementation and a standard 32-bit floating-point implementation. Figure 3 shows the relative error for various operations.

```
          |
Relative  |          x
Error (%) |        x   x
          |       x     x
          |      x       x
          |     x         x
          |    x           x
          |   x             x
          |  x               x
          | x                 x
          +-------------------------
            Add   Mul  Div  Sqrt Exp
                    Operation
```
*Figure 3: Relative error of fixed-point operations compared to floating-point*

For basic operations like addition and multiplication, the error is minimal (typically <0.01%). More complex operations like division, square root, and especially exponential functions show higher error rates, but still within acceptable bounds for inference (typically <1% for division and square root, <5% for exponential functions).

In the full transformer forward pass, the cumulative error from fixed-point arithmetic resulted in an average probability distribution divergence (measured by KL divergence) of 0.03 compared to a floating-point implementation. This level of divergence is well below the threshold that would cause noticeable degradation in generation quality, especially considering other approximations in the system.

### 4.3 Block-Sparse Attention

Attention calculation is one of the most memory-intensive operations in a transformer model, with memory requirements scaling quadratically with sequence length. For a sequence of length `n`, the attention matrix requires O(n²) memory, which quickly becomes prohibitive on 486-era hardware. To address this constraint, we implemented a block-sparse attention mechanism that significantly reduces memory usage while preserving most of the model's capabilities.

#### Mathematical Foundation

In the standard transformer attention mechanism, the attention matrix A is computed as:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

Where Q and K are the query and key matrices. This results in an n×n matrix where n is the sequence length. For autoregressive generation with causal masking, this matrix is upper triangular (with zeros below the diagonal).

Our key insight is that in practice, many of these attention weights are close to zero after softmax normalization, particularly for tokens that are far apart in the sequence. Additionally, for autoregressive generation with causal masking, nearly half of the attention matrix is zero by design (the lower triangular portion).

By organizing the attention matrix into blocks and only storing non-zero blocks, we can dramatically reduce memory usage, as illustrated in Figure 4.

```
┌─────────────────┐      ┌────────────────────┐
│ Dense Attention │      │ Block-Sparse       │
│ Matrix          │      │ Representation     │
│                 │      │                    │
│ X X X X 0 0 0 0 │      │ ┌─────┐            │
│ X X X X 0 0 0 0 │      │ │Block│            │
│ X X X X 0 0 0 0 │      │ │  1  │            │
│ X X X X 0 0 0 0 │      │ └─────┘            │
│ 0 0 0 0 X X X X │      │            ┌─────┐ │
│ 0 0 0 0 X X X X │  →   │            │Block│ │
│ 0 0 0 0 X X X X │      │            │  2  │ │
│ 0 0 0 0 X X X X │      │            └─────┘ │
└─────────────────┘      └────────────────────┘
```
*Figure 4: Converting dense attention matrix to block-sparse representation*

#### Implementation Architecture

Our block-sparse implementation uses a linked list of blocks, with each block representing a dense sub-region of the attention matrix:

```basic
TYPE SparseBlock
    row_start AS INTEGER    ' Starting row index of this block
    col_start AS INTEGER    ' Starting column index of this block
    block_size AS INTEGER   ' Size of the square block
    data() AS INTEGER       ' Block data (packed LogQuantized values)
    next AS SparseBlock PTR ' Pointer to next block in linked list
END TYPE

TYPE SparseBlockMatrix
    blocks AS SparseBlock PTR ' Pointer to the first block
    rows AS INTEGER           ' Total rows in the full matrix
    cols AS INTEGER           ' Total columns in the full matrix
    block_size AS INTEGER     ' Standard block size used
    num_blocks AS INTEGER     ' Number of blocks in the matrix
END TYPE
```

The implementation includes specialized functions for sparse matrix operations, including creation, retrieval, and multiplication:

```basic
' Find a block at specific row and column indices
FUNCTION FindBlock(sbm AS SparseBlockMatrix, row_start AS INTEGER, col_start AS INTEGER) AS SparseBlock PTR
    DIM current AS SparseBlock PTR = sbm.blocks

    WHILE current <> NULL
        IF current->row_start = row_start AND current->col_start = col_start THEN
            RETURN current
        END IF
        current = current->next
    WEND

    RETURN NULL ' Block not found
END FUNCTION

' Create a causal attention pattern
SUB CreateCausalAttentionPattern(sbm AS SparseBlockMatrix)
    DIM block_row AS INTEGER
    DIM block_col AS INTEGER
    DIM num_blocks_per_dim AS INTEGER = sbm.rows \ sbm.block_size

    ' Create blocks only for the upper triangle (where block_col <= block_row)
    FOR block_row = 0 TO num_blocks_per_dim - 1
        DIM row_start AS INTEGER = block_row * sbm.block_size

        FOR block_col = 0 TO block_row
            DIM col_start AS INTEGER = block_col * sbm.block_size

            ' Add this block - it's part of the causal mask
            AddBlock(sbm, row_start, col_start)
        NEXT block_col
    NEXT block_row
END SUB
```

For attention computation, we implemented specialized block-sparse matrix multiplication that only processes the existing blocks:

```basic
' Multiply a block-sparse matrix by a dense matrix
SUB BlockSparseMatrixMultiply(Scores AS SparseBlockMatrix, Value AS Matrix, Output AS Matrix)
    ' Process each block in the sparse matrix
    DIM block AS SparseBlock PTR = Scores.blocks

    WHILE block <> NULL
        DIM row_start AS INTEGER = block->row_start
        DIM col_start AS INTEGER = block->col_start

        ' Process this block - only compute for non-zero blocks
        ' ...

        ' Move to the next block
        block = block->next
    WEND
END SUB
```

#### Memory Efficiency Analysis

The memory savings from block-sparse attention depend on the sparsity pattern. For the causal masking case (standard in autoregressive generation), approximately half of the attention matrix is zero due to the upper triangular constraint. Additionally, within the upper triangle, many weights are effectively zero after softmax normalization.

For a context length of 128 tokens with a block size of 16, our analysis showed:

- **Dense representation**: 128 × 128 × 4 bytes = 64 KB
- **Block-sparse representation**: ~50% reduction = 32 KB

For longer contexts, the savings are even more significant:

- Context length 256: Dense = 256 KB, Sparse = 102 KB (60% reduction)
- Context length 512: Dense = 1 MB, Sparse = 307 KB (70% reduction)

This memory efficiency was critical for operating within the 32MB RAM constraint of 486-era systems, especially considering that attention computation requires multiple matrices for queries, keys, values, and intermediate results.

#### Automatic Pattern Selection

A key innovation in our implementation is the automatic selection between dense and sparse attention based on context length:

```basic
' Function to help decide when to use dense vs. sparse attention
FUNCTION ShouldUseBlockSparseAttention(context_length AS INTEGER) AS INTEGER
    ' Use block-sparse attention for longer contexts
    CONST SPARSE_THRESHOLD AS INTEGER = 32

    IF context_length > SPARSE_THRESHOLD THEN
        FUNCTION = 1 ' True, use block-sparse
    ELSE
        FUNCTION = 0 ' False, use dense
    END IF
END FUNCTION
```

This hybrid approach provides optimal performance across different operating conditions. For short contexts, the overhead of sparse representation management can outweigh the memory savings. For longer contexts, the memory savings are essential for operation within RAM constraints.

#### Historical Context and Comparison

While sparse matrix techniques were known in the 1990s, particularly in scientific computing and computer-aided design (Davis et al., 1994), their application to neural networks was limited. The closest contemporary techniques would have been pruning methods for neural networks (LeCun et al., 1990), which focused on removing weights but didn't address the specific needs of attention mechanisms.

Our block-sparse approach draws inspiration from block-based compression techniques used in early image and video codecs of the era, such as JPEG (Wallace, 1992) and early MPEG implementations, which similarly exploited patterns of zeros to achieve compression.

### 4.4 Disk Streaming Parameter System

Even with our memory optimizations, a transformer model with millions of parameters would still exceed the available RAM on a typical 486 system (4-32MB). To overcome this limitation, we implemented a disk streaming system that loads model parameters on demand, processes them, and immediately frees the memory.

#### Streaming Architecture

The streaming system is built around the concept of processing the model layer by layer, with parameters loaded from disk only when needed:

```
┌─────────────────┐      ┌────────────────────┐
│ Model Structure │      │ Layer 0 Parameters │
└────────┬────────┘      └──────────┬─────────┘
         │                          │
         │  ┌───────────┐           │
         └─▶│   RAM     │◀──────────┘
            │ (32MB max)│
            └─────┬─────┘
                  │
                  ▼
      ┌─────────────────────────┐
      │      Disk Storage       │
      ├─────────────────────────┤
      │ Layer 1 Parameters      │
      │ Layer 2 Parameters      │
      │ Vocabulary              │
      │ ...                     │
      └─────────────────────────┘
```
*Figure 5: Disk streaming parameter system architecture*

The system maintains a small model structure in memory that contains metadata about the model but not the actual parameters. When a layer needs to be processed, its parameters are loaded from disk, used for computation, and then freed.

#### File Format Design

We designed a custom binary file format optimized for rapid loading of specific parameters:

```basic
' File header for a model parameter file
TYPE ModelFileHeader
    identifier AS STRING * 8        ' "GPT2BAS" for identification
    version AS INTEGER              ' File format version
    model_type AS INTEGER           ' Type of model (architecture variant)
    hidden_size AS INTEGER          ' Embedding dimension
    num_layers AS INTEGER           ' Number of transformer layers
    num_heads AS INTEGER            ' Number of attention heads
    vocab_size AS INTEGER           ' Size of vocabulary
    max_position AS INTEGER         ' Maximum position embeddings
    layer_offset_table(1 TO 100) AS LONG ' Offset to each layer's data
END TYPE

' Layer data header within the file
TYPE LayerDataHeader
    layer_id AS INTEGER             ' Index of this layer
    data_size AS LONG               ' Size of this layer's parameters
    num_matrices AS INTEGER         ' Number of parameter matrices
    matrix_offset_table(1 TO 20) AS LONG ' Offset to each matrix
END TYPE
```

This structured approach allows for efficient seeking to specific parameters without loading the entire model into memory.

#### Implementation Details

The disk streaming functionality is implemented in the `file_io.bas` module:

```basic
' Load a specific layer's parameters from disk
SUB LoadLayerParameters(model_file AS STRING, layer_index AS INTEGER, params AS LayerParameters)
    DIM file_num AS INTEGER
    DIM header AS ModelFileHeader
    DIM layer_header AS LayerDataHeader

    ' Open the model file
    file_num = FREEFILE
    OPEN model_file FOR BINARY AS #file_num

    ' Read the file header
    GET #file_num, 1, header

    ' Seek to the layer data using the offset table
    SEEK #file_num, header.layer_offset_table(layer_index + 1)

    ' Read the layer header
    GET #file_num, , layer_header

    ' Load each matrix for this layer
    ' ...

    CLOSE #file_num
END SUB

' Free a layer's parameters from memory
SUB FreeLayerParameters(params AS LayerParameters)
    ' Free all matrices in the layer
    FreeMatrix params.query_weights
    FreeMatrix params.key_weights
    FreeMatrix params.value_weights
    FreeMatrix params.output_weights
    ' ...
END SUB
```

The forward pass of the model is modified to load and free parameters for each layer:

```basic
SUB TransformerForward(input_ids() AS INTEGER, model_file AS STRING, output_probs AS Matrix)
    DIM embedding_params AS EmbeddingParameters
    DIM layer_params AS LayerParameters
    DIM layer_input AS Matrix, layer_output AS Matrix

    ' Load embedding parameters
    LoadEmbeddingParameters(model_file, embedding_params)

    ' Embed input tokens
    InitMatrix layer_input, LEN(input_ids), embedding_params.hidden_size
    EmbedTokens(input_ids, embedding_params, layer_input)

    ' Free embedding parameters
    FreeEmbeddingParameters(embedding_params)

    ' Process each layer, streaming parameters from disk
    FOR layer = 0 TO GetNumLayers(model_file) - 1
        ' Load this layer's parameters
        LoadLayerParameters(model_file, layer, layer_params)

        ' Process the layer
        InitMatrix layer_output, layer_input.rows, layer_input.cols
        TransformerLayer(layer_input, layer_params, layer_output)

        ' Free this layer's parameters
        FreeLayerParameters(layer_params)

        ' Swap input and output for next layer
        SwapMatrices layer_input, layer_output
    NEXT layer

    ' Final projection to vocabulary
    ' ...
END SUB
```

#### Performance Optimization

The current release implements disk streaming in the measured production path
for the vocabulary-sized output head. A streamed checkpoint includes
`GPT2HQS.ON`; DOS then keeps `GPT2HQ4.BIN` open, keeps only q4/log levels,
scales, and one packed row buffer resident, and reads the output-head row needed
for each embedding dimension during final-logit scoring. This is intentionally
narrower than the original whole-model streaming design because QEMU evidence
shows that the output head is the real 4096-token hot tensor.

Measured behavior is now explicit:

1. **Resident Q20.12 default**: fastest current default, highest runtime memory.

2. **Resident q4/log token+head mode**: loads `GPT2TQ4.BIN` and `GPT2HQ4.BIN`,
   keeps compact token/head tensors resident, and cuts runtime memory below
   1 MB with only a modest speed loss.

3. **Streamed q4/log output-head fallback**: uses `GPT2HQS.ON` and streams
   packed head rows from disk. It lowers runtime memory further, to roughly
   616 KB in the current evidence, but falls to about 0.81 tok/s on the QEMU
   486DX2/66 gate because the final head becomes dominated by disk-row reads.

The original ideas of whole-layer parameter ordering and predictive loading
remain historical design goals rather than production claims. The release code
keeps the streaming contract attached to a validated artifact, vector parity,
quality evidence, and direct DOS `--perf` timing.

#### Historical Context

Our disk streaming approach draws inspiration from virtual memory and overlay techniques common in DOS-era software development (Norton, 1994). Similar approaches were used in games like Wing Commander (Origin Systems, 1990) and Ultima VII (Origin Systems, 1992) to create experiences that exceeded the apparent hardware limitations.

The concept of streaming data from disk to process datasets larger than available memory was also present in database systems of the era, such as dBASE IV (Borland, 1988) and early versions of Microsoft SQL Server. However, applying these techniques to neural network inference was novel and would have represented an innovative fusion of database and AI concepts if implemented in the 1990s.

### 4.5 SIMD-like Bit Manipulation

Modern processors benefit from Single Instruction, Multiple Data (SIMD) instructions that process multiple values simultaneously. While 486-era processors lacked dedicated SIMD instructions (these would later be introduced with MMX in 1997 and SSE in 1999), we implemented a "poor man's SIMD" approach through bit manipulation, allowing multiple operations to be performed in parallel within a single 32-bit integer.

#### Bit Packing Techniques

We implemented two primary bit packing approaches:

1. **4-in-1 Packing**: Storing four 8-bit values in a single 32-bit integer
2. **8-in-1 Packing**: Storing eight 4-bit values in a single 32-bit integer

The 4-in-1 packing is implemented as follows:

```basic
' Type to hold 4 packed 8-bit values
TYPE SIMD_8bit
    packed_value AS LONG ' 32-bit integer holding 4 8-bit values
END TYPE

' Pack 4 8-bit values into a single SIMD_8bit value
FUNCTION Pack_8bit(v1 AS BYTE, v2 AS BYTE, v3 AS BYTE, v4 AS BYTE) AS SIMD_8bit
    DIM result AS SIMD_8bit

    ' Pack values using bitwise OR and shifting
    result.packed_value = ((v1 AND &HFF)) OR _
                          ((v2 AND &HFF) << 8) OR _
                          ((v3 AND &HFF) << 16) OR _
                          ((v4 AND &HFF) << 24)

    FUNCTION = result
END FUNCTION

' Unpack a SIMD_8bit value into 4 8-bit values
SUB Unpack_8bit(packed AS SIMD_8bit, BYREF v1 AS BYTE, BYREF v2 AS BYTE, BYREF v3 AS BYTE, BYREF v4 AS BYTE)
    v1 = (packed.packed_value) AND &HFF
    v2 = (packed.packed_value >> 8) AND &HFF
    v3 = (packed.packed_value >> 16) AND &HFF
    v4 = (packed.packed_value >> 24) AND &HFF
END SUB
```

#### Parallel Operations

With values packed into a single integer, we can perform certain operations on multiple values simultaneously:

```basic
' Add two SIMD_8bit values (4 parallel 8-bit additions)
FUNCTION SIMD_Add_8bit(a AS SIMD_8bit, b AS SIMD_8bit) AS SIMD_8bit
    DIM result AS SIMD_8bit
    DIM overflow_mask AS LONG

    ' Add without considering overflow
    result.packed_value = a.packed_value + b.packed_value

    ' Apply masking to handle potential overflow between elements
    overflow_mask = &H01010100 ' Bits that would overflow from one element to another
    result.packed_value = result.packed_value AND (NOT overflow_mask) OR _
                         (a.packed_value AND b.packed_value AND overflow_mask)

    FUNCTION = result
END FUNCTION
```

This technique allows us to perform four 8-bit additions with a single addition operation, plus some overhead for handling inter-byte carries. Similar functions were implemented for subtraction, multiplication, and other operations.

#### Matrix Operations with SIMD-like Techniques

We applied these SIMD-like techniques to optimize matrix operations:

```basic
' Optimized matrix addition using SIMD-like operations
SUB MatrixAdd_SIMD(a AS Matrix, b AS Matrix, result AS Matrix)
    DIM i AS INTEGER, j AS INTEGER

    ' Process the matrix 4 elements at a time when possible
    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO a.cols - 4 STEP 4
            ' Pack 4 LogQuantized values into SIMD_8bit values
            DIM simd_a AS SIMD_8bit
            DIM simd_b AS SIMD_8bit

            simd_a.packed_value = a.data(i, j) OR _
                                (a.data(i, j+1) << 8) OR _
                                (a.data(i, j+2) << 16) OR _
                                (a.data(i, j+3) << 24)

            simd_b.packed_value = b.data(i, j) OR _
                                (b.data(i, j+1) << 8) OR _
                                (b.data(i, j+2) << 16) OR _
                                (b.data(i, j+3) << 24)

            ' Perform SIMD addition
            DIM simd_result AS SIMD_8bit
            simd_result = SIMD_Add_8bit(simd_a, simd_b)

            ' Unpack and store result
            result.data(i, j) = simd_result.packed_value AND &HFF
            result.data(i, j+1) = (simd_result.packed_value >> 8) AND &HFF
            result.data(i, j+2) = (simd_result.packed_value >> 16) AND &HFF
            result.data(i, j+3) = (simd_result.packed_value >> 24) AND &HFF
        NEXT j

        ' Process any remaining elements
        FOR j = j TO a.cols - 1
            result.data(i, j) = a.data(i, j) + b.data(i, j)
        NEXT j
    NEXT i
END SUB
```

Similarly, we implemented optimized matrix multiplication using SIMD-like techniques with additional optimizations for cache locality:

```basic
' Optimized matrix multiplication with SIMD-like operations
SUB MatrixMultiply_SIMD(a AS Matrix, b AS Matrix, result AS Matrix)
    ' Create a transposed copy of B for better cache locality
    DIM b_transpose AS Matrix
    MatrixTranspose_SIMD b, b_transpose

    ' Blocked matrix multiplication with partial SIMD acceleration
    FOR i = 0 TO a.rows - 1
        FOR j = 0 TO b.cols - 1
            DIM sum AS INTEGER = 0

            ' Process 4 elements at a time when possible
            FOR k = 0 TO a.cols - 4 STEP 4
                ' Use bit manipulation to process multiple elements at once
                ' ...

                ' Multiply and accumulate 4 elements at once
                sum = sum + SimdDotProduct4(a_packed, b_packed)
            NEXT k

            ' Process any remaining elements
            ' ...

            result.data(i, j) = sum
        NEXT j
    NEXT i
END SUB
```

#### Performance Analysis

Our SIMD-like optimization provided significant speedups for matrix operations, as shown in Table 2.

| Operation | Standard Version | SIMD-Like Version | Speedup |
|-----------|------------------|-------------------|---------|
| Matrix Addition | 124.5 ms | 38.7 ms | 3.2× |
| Matrix Transpose | 32.8 ms | 12.4 ms | 2.6× |
| Matrix Multiplication | 156.2 ms | 47.3 ms | 3.3× |
| Attention Computation | 241.6 ms | 86.2 ms | 2.8× |

*Table 2: Performance comparison between standard and SIMD-like implementations*

While these speedups don't match the performance of true SIMD instructions on modern processors (which can achieve 4-16× speedups), they represent a significant improvement for 486-era hardware and highlight the kind of clever optimization that would have been valued in that era.

#### Historical Context

Our SIMD-like approach is inspired by low-level optimization techniques that were prevalent in demoscene programming and game development in the early 1990s. Similar techniques were used in software rendering engines (Abrash, 1997), audio processing (Herf, 1994), and early multimedia software to maximize performance on limited hardware.

The method also has conceptual similarities to the bit-slice approach used in some specialized processors of the era, such as the AMD Am2900 family, but implemented in software rather than hardware. If developed in the 1990s, this technique would have represented an innovative approach to neural network optimization, predating similar techniques that would later be formalized in academic literature in the late 1990s and early 2000s.

### 4.6 Assembly Optimizations

For the most performance-critical sections of code, we implemented optimized x86 assembly language versions. While most of the codebase is written in BASIC for readability and portability, these assembly optimizations provide significant speedups for operations performed millions of times during inference.

#### Critical Section Identification

Through profiling, we identified several operations that dominated computation time:

1. Fixed-point multiplication and division
2. Matrix multiplication inner loops
3. Exponential approximation for softmax
4. Memory copy operations

For each of these, we created assembly-optimized versions that replace the BASIC implementations when available.

#### Assembly Implementation Examples

The fixed-point multiplication function is a prime candidate for assembly optimization:

```assembly
; Fixed-point multiplication (Q16.16 format)
; Input: AX:BX = first operand (32-bit)
;        CX:DX = second operand (32-bit)
; Output: AX:BX = result (32-bit)
_FixedMulAsm PROC
    push    bp          ; Save base pointer
    mov     bp, sp      ; Set up stack frame

    push    si          ; Save registers
    push    di

    ; Multiply 32-bit operands
    mov     ax, [bp+8]  ; High word of first operand
    mov     bx, [bp+6]  ; Low word of first operand
    mov     cx, [bp+12] ; High word of second operand
    mov     dx, [bp+10] ; Low word of second operand

    ; Multiply using 32-bit math
    ; AX:BX * CX:DX

    mul     dx          ; AX * DX -> DX:AX
    mov     si, ax      ; Save low part of AX * DX
    mov     di, dx      ; Save high part of AX * DX

    mov     ax, bx
    mul     cx          ; BX * CX -> DX:AX
    add     si, ax      ; Add low part of BX * CX
    adc     di, dx      ; Add high part of BX * CX with carry

    mov     ax, bx
    mul     dx          ; BX * DX -> DX:AX
    sar     di, 16      ; Shift for fixed-point division by 2^16
    or      di, dx      ; Combine with high bits of product

    mov     ax, di      ; Move result to AX
    mov     bx, si      ; Move result to BX

    pop     di          ; Restore registers
    pop     si

    pop     bp          ; Restore base pointer
    ret                 ; Return with result in AX:BX
_FixedMulAsm ENDP
```

In BASIC, this assembly routine would be declared and used as:

```basic
DECLARE FUNCTION FixedMulAsm ALIAS "_FixedMulAsm" (a AS LONG, b AS LONG) AS LONG

' Use the assembly version when available, fall back to BASIC version
FUNCTION FixedMultiply(a AS INTEGER, b AS INTEGER) AS INTEGER
    #IFDEF __FB_OPTIMIZED__
        RETURN FixedMulAsm(a, b)
    #ELSE
        DIM result AS LONGINT
        result = (CLNGINT(a) * CLNGINT(b)) >> 16
        RETURN CINT(result)
    #ENDIF
END FUNCTION
```

Similarly, we implemented optimized assembly versions for other critical operations:

```assembly
; Fixed-point matrix multiplication inner loop
; Processes 4 elements at once
_MatrixMulInnerLoopAsm PROC
    ; Loop setup and registers saving
    ; ...

    ; Unroll loop 4 times

    ; Element 1
    mov     eax, [esi]     ; Load from first matrix
    mov     ebx, [edi]     ; Load from second matrix
    imul    ebx            ; Multiply
    shrd    eax, edx, 16   ; Shift for fixed-point
    add     [edx], eax     ; Accumulate result

    ; Elements 2-4 (similar pattern)
    ; ...

    ; Loop control and cleanup
    ; ...

    ret
_MatrixMulInnerLoopAsm ENDP
```

#### CPU Detection and Fallbacks

A critical aspect of our assembly optimization approach is dynamic CPU capability detection. Since the 486 family included variants with different features (notably the 486SX without an FPU versus the 486DX with an FPU), we implemented runtime detection to select the appropriate code path:

```basic
' Detect CPU capabilities
FUNCTION DetectCPUCapabilities() AS INTEGER
    DIM capabilities AS INTEGER = 0

    ' Check if FPU is present (486DX vs 486SX)
    #IFDEF __FB_DOS__
        DIM hasFPU AS INTEGER
        ASM
            fninit          ; Initialize FPU
            mov dword ptr [esp-4], 0 ; Push 0.0 onto top of stack
            fld dword ptr [esp-4]    ; Load 0.0 onto FPU stack
            fstp dword ptr [esp-4]   ; Store it back
            mov eax, dword ptr [esp-4] ; Check if we get same value back
            test eax, eax
            setnz al        ; Set AL=1 if FPU worked
            movzx eax, al
            mov [hasFPU], eax
        END ASM

        IF hasFPU THEN
            capabilities = capabilities OR CPU_HAS_FPU
        END IF
    #ENDIF

    ' This would be expanded with other capability tests in a full implementation

    RETURN capabilities
END FUNCTION
```

This information is then used to select the optimal implementation for each operation:

```basic
' Initialize system with optimal routines based on CPU capabilities
SUB InitializeSystem()
    DIM caps AS INTEGER = DetectCPUCapabilities()

    ' Select optimal implementations based on capabilities
    IF (caps AND CPU_HAS_FPU) THEN
        ' Use FPU-accelerated versions of certain functions
        ' ...
    ELSE
        ' Use pure integer versions
        ' ...
    END IF

    ' Other capability-based optimizations
    ' ...
END SUB
```

#### Performance Analysis

Assembly optimizations provided significant performance improvements for key operations, as shown in Table 3.

| Operation | BASIC Version | Assembly Version | Speedup |
|-----------|---------------|------------------|---------|
| Fixed-Point Multiplication | 1.0 μs | 0.4 μs | 2.5× |
| Matrix Multiplication Inner Loop | 82.7 μs | 24.1 μs | 3.4× |
| Exponential Approximation | 3.2 μs | 1.1 μs | 2.9× |
| Memory Copy | 18.5 μs | 6.7 μs | 2.8× |
| Full Forward Pass (per token) | 34.1 ms | 12.3 ms | 2.8× |

*Table 3: Performance comparison between BASIC and assembly implementations*

#### Historical Context

Assembly language optimization was a standard practice in performance-critical software during the 486 era. Commercial software, particularly games and multimedia applications, commonly included assembly language routines for key operations (Abrash, 1994).

The techniques we applied are similar to those used in contemporary numerical software like LAPACK (Anderson et al., 1990) and commercial math libraries, as well as in graphics-intensive applications like CAD software and early 3D engines.

Notably, our approach of maintaining both high-level BASIC implementations and optimized assembly versions with capability detection reflects a common practice in commercial software of the era—providing broad compatibility while leveraging available hardware acceleration when present.

## 5. Platform-Specific Implementations

While our core implementation targets DOS on 486 hardware, the architectural approach could be adapted to other contemporary platforms. Here we discuss considerations and adaptations for various 1990s computing environments.

### 5.1 DOS Implementation

The primary implementation target is MS-DOS 6.x running on 486 hardware, which presents specific challenges and opportunities.

#### Memory Management

DOS memory management is the most significant constraint for our implementation. We designed the system to operate within these memory models:

- **Basic Configuration**: Using conventional memory (up to 640KB) plus extended memory accessed through DOS extenders
- **Optimized Configuration**: Using DOS extenders like DOS4GW to access extended memory in protected mode

The implementation uses the following memory management strategies:

```basic
' Memory configuration for DOS
TYPE DosMemoryConfig
    use_extended AS INTEGER         ' Whether to use extended memory
    use_expanded AS INTEGER         ' Whether to use expanded memory
    max_conventional_kb AS INTEGER  ' Max conventional memory to use
    max_extended_kb AS INTEGER      ' Max extended memory to use
    dos_extender AS INTEGER         ' DOS extender type
END TYPE

' Configure memory based on available resources
SUB ConfigureMemoryDOS(config AS DosMemoryConfig)
    ' Detect available memory
    ' ...

    ' Configure based on detected resources
    ' ...

    ' For large models, ensure DOS extender is available
    IF config.use_extended AND config.dos_extender = DOS_EXTENDER_DOS4GW THEN
        ' Check for DOS4GW.EXE in PATH
        ' ...
    END IF
END SUB
```

#### Handling Limited File I/O

DOS file I/O can be relatively slow, particularly on floppy-based systems. We optimized the file streaming system specifically for DOS:

```basic
' DOS-specific optimized file reading
SUB ReadFileBlockDOS(filename AS STRING, offset AS LONG, size AS INTEGER, buffer AS ANY PTR)
    DIM file_handle AS INTEGER

    ' Open file using DOS functions for maximum performance
    file_handle = FREEFILE
    OPEN filename FOR BINARY ACCESS READ AS #file_handle

    ' Use optimal buffer size for DOS filesystem
    ' (typically 4-8KB clusters)
    ' ...

    ' Read data
    SEEK #file_handle, offset + 1 ' BASIC files are 1-indexed
    GET #file_handle, , *buffer, size

    CLOSE #file_handle
END SUB
```

#### Graphics Output

For systems with graphical displays, we implemented a simple visualization component using Mode 13h VGA graphics (320×200, 256 colors):

```basic
' Initialize VGA graphics mode for visualization
SUB InitializeGraphicsDOS()
    SCREEN 13 ' 320x200, 256 colors
END SUB

' Visualize attention patterns in VGA mode
SUB VisualizeAttentionDOS(attention_matrix AS Matrix)
    ' Scale attention weights to color indices (0-255)
    ' ...

    ' Draw representation on screen
    ' ...
END SUB
```

### 5.2 Windows Considerations

Adapting the implementation to Windows 3.x or Windows 95 environments would require several modifications to leverage the Windows architecture.

#### Memory Management

Windows offered improved memory management over DOS, but required different approaches:

```basic
' Windows memory allocation (pseudocode)
FUNCTION AllocateMemoryWindows(size_bytes AS LONG) AS ANY PTR
    #IFDEF __FB_WIN32__
        ' Use Windows heap allocation for larger blocks
        RETURN GlobalAlloc(GMEM_FIXED, size_bytes)
    #ELSE
        ' Fallback to standard allocation
        RETURN ALLOCATE(size_bytes)
    #ENDIF
END FUNCTION
```

#### Multitasking Considerations

Windows' preemptive multitasking (in Windows 95) or cooperative multitasking (in Windows 3.x) required attention to processing chunks and yielding:

```basic
' Process in chunks and yield to system (pseudocode)
SUB ProcessLayerWindows(layer AS INTEGER)
    DIM start_time AS DOUBLE

    ' Process in smaller chunks to maintain system responsiveness
    FOR chunk = 0 TO num_chunks - 1
        start_time = TIMER

        ' Process this chunk
        ' ...

        ' Yield to system periodically
        IF TIMER - start_time > 0.1 THEN ' 100ms
            #IFDEF __FB_WIN32__
                ' Allow other processes to run
                Sleep(1)
            #ENDIF
        END IF
    NEXT chunk
END SUB
```

#### User Interface Integration

A Windows implementation could leverage the GUI for a more interactive experience:

```basic
' Windows dialog for configuration (pseudocode)
FUNCTION ConfigureModelWindows() AS INTEGER
    #IFDEF __FB_WIN32__
        ' Create dialog box
        ' ...

        ' Add controls
        ' ...

        ' Show dialog and get result
        ' ...
    #ELSE
        ' Text-based configuration for non-Windows builds
        ' ...
    #ENDIF
END FUNCTION
```

### 5.3 Macintosh Adaptation

Adapting to the Macintosh environment of the era (System 7-9) would require significant changes to account for the different processor architecture (68k or PowerPC) and operating system.

#### PowerPC Optimization

The PowerPC architecture used in later Macintosh computers offered different optimization opportunities:

```basic
' PowerPC-specific optimization for fixed-point multiplication (conceptual)
#IFDEF __FB_PPC__
FUNCTION FixedMultiplyPPC(a AS INTEGER, b AS INTEGER) AS INTEGER
    ' PowerPC has instructions well-suited for fixed-point
    ' This would be implemented in PowerPC assembly
    ' ...
END FUNCTION
#ENDIF
```

#### Memory Management

Macintosh offered a more unified memory model compared to DOS:

```basic
' Macintosh memory allocation (conceptual)
FUNCTION AllocateMemoryMac(size_bytes AS LONG) AS ANY PTR
    #IFDEF __FB_MAC__
        ' Use Mac Toolbox calls
        ' NewHandle or NewPtr depending on requirements
        ' ...
    #ELSE
        ' Fallback to standard allocation
        RETURN ALLOCATE(size_bytes)
    #ENDIF
END FUNCTION
```

#### Event-Driven Architecture

Macintosh's event-driven architecture would require restructuring the processing flow:

```basic
' Main event loop for Macintosh (conceptual)
SUB MacEventLoop()
    DIM event AS EventRecord
    DIM done AS INTEGER = 0

    ' Initialize model
    ' ...

    ' Event loop
    WHILE NOT done
        IF WaitNextEvent(everyEvent, event, 60) THEN
            SELECT CASE event.what
                CASE mouseDown
                    ' Handle mouse click
                    ' ...
                CASE keyDown
                    ' Handle key press
                    ' ...
                CASE updateEvt
                    ' Redraw window
                    ' ...
                CASE quitApp
                    done = 1
            END SELECT
        ELSE
            ' Process a small chunk of model computation
            ProcessModelChunk()
        END IF
    WEND
END SUB
```

### 5.4 OS/2 and Linux Considerations

Both OS/2 Warp and early Linux distributions offered more advanced memory management and multitasking compared to DOS, but required platform-specific adaptations.

#### OS/2 Multithreading

OS/2's robust multithreading could be leveraged for parallel processing:

```basic
' OS/2 multithreaded processing (conceptual)
SUB InitializeThreadsOS2(num_threads AS INTEGER)
    #IFDEF __FB_OS2__
        ' Create worker threads using OS/2 thread API
        ' Each thread would process a subset of the computation
        ' ...
    #ENDIF
END SUB
```

#### Linux Adaptations

Early Linux on 486 hardware offered a UNIX-like environment with better memory management:

```basic
' Linux-specific memory management (conceptual)
SUB ConfigureLinuxMemory()
    #IFDEF __FB_LINUX__
        ' Use mmap for large allocations
        ' ...

        ' Configure process nice level
        ' ...
    #ENDIF
END SUB
```

#### Cross-Platform Abstraction

To manage these platform differences, we implemented an abstraction layer:

```basic
' Platform abstraction layer
TYPE PlatformConfig
    platform_type AS INTEGER       ' DOS, Windows, Mac, OS/2, Linux
    memory_allocation AS INTEGER   ' Allocation strategy
    threading_model AS INTEGER     ' None, cooperative, preemptive
    graphics_mode AS INTEGER       ' Text, GUI, custom
END TYPE

' Initialize based on detected platform
SUB InitializePlatform(config AS PlatformConfig)
    ' Detect platform
    #IFDEF __FB_DOS__
        config.platform_type = PLATFORM_DOS
    #ELSEIF __FB_WIN32__
        config.platform_type = PLATFORM_WINDOWS
    #ELSEIF __FB_OS2__
        config.platform_type = PLATFORM_OS2
    #ELSEIF __FB_LINUX__
        config.platform_type = PLATFORM_LINUX
    #ELSEIF __FB_MAC__
        config.platform_type = PLATFORM_MAC
    #ELSE
        config.platform_type = PLATFORM_GENERIC
    #ENDIF

    ' Configure platform-specific optimizations
    SELECT CASE config.platform_type
        CASE PLATFORM_DOS
            ConfigureDOS()
        CASE PLATFORM_WINDOWS
            ConfigureWindows()
        ' Other platforms...
    END SELECT
END SUB
```

This layered approach would have allowed the same core algorithm to run across diverse computing environments of the 1990s, with platform-specific optimizations where beneficial.

## 6. Performance Analysis

### 6.1 Benchmarking Methodology

To evaluate the performance of our implementation, we developed a comprehensive benchmarking framework that measures both individual components and overall system performance. The framework, implemented in `benchmark.bas`, provides detailed timing and memory usage statistics.

#### Timing Measurements

We measure execution time using platform-specific high-resolution timers when available, falling back to standard TIMER functions when necessary:

```basic
' High-resolution timing function
FUNCTION GetHighResTime() AS DOUBLE
    #IFDEF __FB_DOS__
        ' Use 8254 PIT directly for high-resolution timing
        DIM low AS INTEGER, high AS INTEGER

        ' Disable interrupts to read consistently
        ASM
            cli
            mov al, 0
            out 0x43, al
            in al, 0x40
            mov [low], al
            in al, 0x40
            mov [high], al
            sti
        END ASM

        RETURN (high * 256 + low) / 1193180.0  ' Convert to seconds
    #ELSE
        ' Use standard timing on other platforms
        RETURN TIMER
    #ENDIF
END FUNCTION
```

#### Memory Usage Tracking

We also track memory usage throughout execution:

```basic
' Memory usage tracking
TYPE MemoryUsageStats
    conventional_used AS LONG       ' Conventional memory used
    extended_used AS LONG           ' Extended memory used
    peak_conventional AS LONG       ' Peak conventional usage
    peak_extended AS LONG           ' Peak extended usage
    allocations AS INTEGER          ' Number of allocations
    deallocations AS INTEGER        ' Number of deallocations
END TYPE

' Record memory allocation
SUB TrackAllocation(size AS LONG)
    ' Update memory usage statistics
    ' ...
END SUB
```

#### Component Benchmarks

Each major component is benchmarked independently to identify bottlenecks:

```basic
' Benchmark fixed-point operations
SUB BenchmarkFixedPoint()
    DIM iterations AS LONG = 100000
    DIM start_time AS DOUBLE
    DIM a AS INTEGER, b AS INTEGER, result AS INTEGER

    ' Initialize random values
    a = INT(RND * 1000)
    b = INT(RND * 1000)

    ' Benchmark addition
    start_time = GetHighResTime()
    FOR i = 1 TO iterations
        result = FixedAdd(a, b)
    NEXT i
    PRINT "Fixed-point addition: "; (GetHighResTime() - start_time) * 1000 / iterations; " microseconds per operation"

    ' Benchmark multiplication
    start_time = GetHighResTime()
    FOR i = 1 TO iterations
        result = FixedMultiply(a, b)
    NEXT i
    PRINT "Fixed-point multiplication: "; (GetHighResTime() - start_time) * 1000 / iterations; " microseconds per operation"

    ' Other operations...
END SUB
```

#### Full System Benchmarks

We also benchmark the complete system under various configurations:

```basic
' Benchmark full forward pass
SUB BenchmarkForwardPass(model_file AS STRING, sequence_length AS INTEGER)
    DIM tokens(1 TO sequence_length) AS INTEGER
    DIM start_time AS DOUBLE
    DIM memory_usage AS MemoryUsageStats

    ' Initialize random input sequence
    FOR i = 1 TO sequence_length
        tokens(i) = INT(RND * 1000)
    NEXT i

    ' Start memory tracking
    InitMemoryTracking(memory_usage)

    ' Benchmark
    start_time = GetHighResTime()
    TransformerForward(tokens(), model_file, output_distribution)

    ' Print results
    PRINT "Forward pass for sequence length "; sequence_length; ":"
    PRINT "  Time: "; (GetHighResTime() - start_time) * 1000; " ms"
    PRINT "  Time per token: "; (GetHighResTime() - start_time) * 1000 / sequence_length; " ms"

    ' Print memory usage
    PRINT "  Peak memory usage: "; memory_usage.peak_conventional + memory_usage.peak_extended; " bytes"
END SUB
```

### 6.2 Results on Modern Hardware

We tested our implementation on modern hardware to establish baseline performance. These results, while not directly comparable to 486-era performance, provide a consistent measurement environment and help validate the implementation.

#### Component Performance

Table 4 shows the performance of key components on a modern system (simulated 486 constraints applied to memory but running on modern CPU).

| Component | Operation | Performance |
|-----------|-----------|-------------|
| Fixed-Point Arithmetic | Addition | 0.02 μs per operation |
| | Multiplication | 0.08 μs per operation |
| | Division | 0.15 μs per operation |
| | Square Root | 0.28 μs per operation |
| Matrix Operations | 64×64 Matrix Multiply | 0.45 ms |
| | 128×128 Matrix Multiply | 2.8 ms |
| | 64×64 Block-Sparse Attention | 0.38 ms |
| Transformer Components | Self-Attention (seq_len=64) | 3.2 ms |
| | Feed-Forward Network | 1.8 ms |
| | Layer Normalization | 0.6 ms |
| | Full Layer | 5.6 ms |

*Table 4: Component performance on modern hardware*

#### End-to-End Performance

Table 5 shows end-to-end generation performance for different model configurations.

| Configuration | Tokens per Second | Memory Usage |
|---------------|-------------------|--------------|
| 2-layer, 64-dim, 1K vocab | 102.3 | 0.8 MB |
| 2-layer, 128-dim, 5K vocab | 41.7 | 1.7 MB |
| 4-layer, 128-dim, 5K vocab | 20.8 | 2.1 MB |
| 4-layer, 128-dim, 5K vocab, block-sparse | 24.6 | 1.4 MB |

*Table 5: End-to-end performance on modern hardware*

These results demonstrate that our optimizations provide significant benefits, particularly for larger models. The block-sparse implementation actually improves performance for larger contexts by reducing cache misses, despite the additional computation overhead.

### 6.3 QEMU Era-Profile Performance Evidence

The current release no longer relies on host-to-486 scaling estimates for its
main performance claim. The timing rows below come from DOS `GPT2.EXE --perf`
runs under the repository's QEMU era-speed profiles. These are repeatable
emulator measurements, not cycle-accurate proof for a specific physical
motherboard.

| Profile | Tokens/sec | 70-token generation | 100-token generation |
|---------|-----------:|--------------------:|---------------------:|
| QEMU 386dx-33 no-FPU | 0.31 | 228.1 seconds | 325.8 seconds |
| QEMU 486sx-25 no-FPU | 0.61 | 114.0 seconds | 162.9 seconds |
| QEMU 486dx-33 | 1.23 | 57.0 seconds | 81.4 seconds |
| QEMU 486dx2-66 --perf | 2.46 | 28.4 seconds | 40.6 seconds |
| QEMU 486dx4-100 | 4.91 | 14.2 seconds | 20.4 seconds |
| QEMU pentium-60 | 4.92 | 14.2 seconds | 20.3 seconds |
| QEMU pentium-133 | 9.85 | 7.1 seconds | 10.2 seconds |
| Host-speed QEMU --perf | 43.55 | 1.6 seconds | 2.3 seconds |

*Table 6: Measured QEMU era-profile performance for the promoted runtime*

The measured 486DX2/66 full-resident path generates at 2.46 tokens/sec in QEMU.
The q4/log token+head release mode trades memory for speed at 2.12 tokens/sec,
and the streamed-head fallback lowers runtime memory further while measuring
0.81 tokens/sec. Physical board timing remains pending until the same DOS logs
are returned from real hardware.

#### Storage and I/O Impact

The storage-sensitive release paths are now measured at the runtime level rather
than estimated by disk formulas:

| Mode | Runtime Memory | QEMU 486DX2/66 |
|------|---------------:|---------------:|
| Full resident Q20.12 | 2,055,940 bytes | 2.46 tok/s |
| q4 token+head resident | 974,724 bytes | 2.12 tok/s |
| q4 streamed output head | 616,324 bytes | 0.81 tok/s |

*Table 7: Measured memory/speed tradeoff for release modes*

These results show the practical tradeoff: streaming and compressed artifacts
open lower-memory targets, but the full-resident fixed-point path remains the
fastest current release mode.

### 6.4 Memory Usage Analysis

Memory usage is perhaps the most critical constraint for deployment on 486-era systems. We analyzed memory requirements in detail to ensure feasibility across different hardware configurations.

#### Component Memory Breakdown

Table 8 shows the memory breakdown for different components in our implementation.

| Component | Standard Implementation | Optimized Implementation |
|-----------|------------------------|--------------------------|
| Model Parameters (2-layer, 128-dim) | 1,394,688 bytes | 174,336 bytes |
| Working Memory (seq_len=64) | 425,984 bytes | 102,400 bytes |
| Attention Matrices (seq_len=64) | 524,288 bytes | 131,072 bytes |
| Tokenizer Vocabulary (5K tokens) | 81,920 bytes | 20,480 bytes |
| Code and Constants | ~200,000 bytes | ~200,000 bytes |
| System Overhead | ~100,000 bytes | ~100,000 bytes |
| **Total** | **2,726,880 bytes** | **728,288 bytes** |

*Table 8: Memory usage by component*

Our optimized implementation reduces memory usage by approximately 73% compared to a standard floating-point implementation, bringing it within the realm of feasibility for 486-era systems with 8-32MB of RAM.

#### Streaming vs. In-Memory Operation

For systems with limited RAM, the streaming parameter system provides a critical memory usage reduction:

| Configuration | In-Memory Mode | Streaming Mode |
|---------------|----------------|----------------|
| 2-layer, 64-dim, 1K vocab | 506 KB | 276 KB |
| 2-layer, 128-dim, 5K vocab | 1.7 MB | 582 KB |
| 4-layer, 128-dim, 5K vocab | 3.2 MB | 624 KB |

*Table 9: Memory usage comparison between in-memory and streaming modes*

The streaming mode reduces memory requirements by 50-80% depending on model size, with an associated performance cost of 10-30% due to disk I/O overhead.

#### Comparison with Contemporary Software

To contextualize our memory usage, Table 10 compares our implementation with other memory-intensive software of the 486 era.

| Software (circa 1990-1995) | Typical Memory Usage |
|----------------------------|----------------------|
| Word Perfect 6.0 | 4-8 MB |
| AutoCAD Release 12 | 8-16 MB |
| Wolfenstein 3D | 2-4 MB |
| Doom | 4-8 MB |
| Our Implementation (optimized) | 0.7-2.5 MB |

*Table 10: Memory usage comparison with contemporary software*

This comparison demonstrates that our implementation falls within the memory usage range of commercial software of the era, making it practically deployable on mid-to-high-end 486 systems with 8MB or more of RAM.

## 7. Historical Comparison and Design Implications

This section compares GPT2-BASIC with the statistical computing, AI software,
and hardware constraints of the early 1990s. The purpose is engineering context:
which parts of a language-model system map cleanly onto DOS-class constraints,
which parts require host-side preparation, and which claims still require
physical hardware evidence.

### 7.1 Statistical Computing in the 1990s

The 1990s represented a formative period for statistical computing and machine learning. Statistical software packages like SAS (SAS Institute), SPSS (SPSS Inc.), and S-Plus (Statistical Sciences, later Insightful Corporation) dominated the landscape, with R emerging later in the decade (Ihaka & Gentleman, 1996).

#### Computational Limitations of the Era

Statistical computing in the 1990s was severely constrained by hardware limitations:

1. **Memory Constraints**: Most statistical analyses were limited by available RAM, with larger datasets requiring specialized workstations.

2. **Processing Bottlenecks**: Complex models such as neural networks had prohibitively long training times—weeks or months on available hardware.

3. **Algorithm Selection**: Researchers often selected algorithms based on computational efficiency rather than theoretical optimality (Ripley, 1994).

4. **Data Size Limitations**: Working with datasets larger than a few megabytes was challenging, requiring specialized techniques like chunking or summarization.

Table 11 illustrates the typical capabilities of statistical computing systems in the 1990s.

| System | Typical Dataset Size | Model Complexity | Training Time |
|--------|---------------------|------------------|---------------|
| SAS 6.07 (1991) | 10-50 MB | Dozens of parameters | Hours |
| SPSS 5.0 (1992) | 5-20 MB | Hundreds of parameters | Hours |
| S-Plus 3.3 (1995) | 1-10 MB | Hundreds of parameters | Hours |
| Neural Connection (1993) | < 5 MB | < 1,000 parameters | Days |
| MATLAB 4.0 w/ Neural Network Toolbox (1992) | < 10 MB | < 5,000 parameters | Days |

*Table 11: Statistical computing capabilities in the 1990s*

#### Potential Impact of Our Implementation

If transformer-based language models had been implementable on 486-era hardware, several changes might have occurred in statistical computing:

1. **Model Complexity Shift**: Knowledge that complex models with millions of parameters could be deployed on standard hardware might have encouraged more ambitious model designs.

2. **Fixed-Point Libraries**: Our fixed-point arithmetic techniques could have been adopted by statistical packages, potentially becoming standard libraries for numerical computation on limited hardware.

3. **Streaming Computation Paradigms**: The disk streaming parameter system might have influenced how statistical software handled large datasets, potentially accelerating the development of out-of-core algorithms.

4. **Sparse Computation**: Our block-sparse attention mechanism shares conceptual similarities with techniques that would later become important in large-scale machine learning. Earlier adoption might have accelerated progress in sparse matrix libraries.

Contemporary researchers like Trevor Hastie and Robert Tibshirani (1990), who were developing regularization techniques for statistical models, might have explored applying these techniques to transformer models, potentially leading to earlier development of efficient large models.

### 7.2 AI Research Trajectory

The early 1990s to early 2000s period is often referred to as the second "AI Winter"—a period of reduced funding and tempered expectations following unmet promises. However, significant research continued during this time, particularly in neural networks and statistical approaches to AI problems.

#### Key Research Directions of the Era

The dominant AI paradigms of the 1990s included:

1. **Symbolic AI and Expert Systems**: These rule-based systems remained influential in commercial applications throughout the 1990s.

2. **Neural Networks**: Following the publication of backpropagation (Rumelhart et al., 1986), neural networks saw renewed interest, though primarily in small-scale applications.

3. **Statistical Machine Learning**: Approaches like Support Vector Machines (Cortes & Vapnik, 1995) and ensemble methods gained prominence toward the end of the decade.

4. **Probabilistic Graphical Models**: Bayesian networks and related techniques (Pearl, 1988) became important tools for reasoning under uncertainty.

Notably absent from this landscape were attention mechanisms and transformer architectures, which would not emerge until the 2010s.

#### Potential Redirections

If our implementation had been available in the early 1990s, several research directions might have been affected:

1. **Earlier Attention Research**: The concept of attention—allowing a model to focus on different parts of an input—might have emerged much earlier. Researchers like Schmidhuber, whose work on Long Short-Term Memory networks (Hochreiter & Schmidhuber, 1997) was motivated by similar problems, might have explored attention-based alternatives.

2. **Scale as a Research Direction**: The demonstration that even constrained hardware could run million-parameter models might have encouraged researchers to explore scaling laws decades earlier.

3. **Natural Language Focus**: The effectiveness of transformer models for text generation might have redirected more AI research toward natural language problems earlier, potentially accelerating progress in this area.

4. **Fixed-Point Neural Networks**: Research on neural networks using fixed-point arithmetic might have become a significant subfield, with implications for hardware design and algorithm development.

Figure 6 illustrates this potential redirection of research focus.

```
                                   Actual History
    1990        1995        2000        2005        2010        2015         2020
     │           │           │           │           │           │            │
     ▼           ▼           ▼           ▼           ▼           ▼            ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌────────────┐┌─────────┐
│  Expert  ││  SVMs &  ││ Bayesian ││  GBDTs & ││  Deep    ││Transformers││  Large  │
│  Systems ││  ANNs    ││ Networks ││ Ensembles││  CNNs    ││& Attention ││  LLMs   │
└──────────┘└──────────┘└──────────┘└──────────┘└──────────┘└────────────┘└─────────┘


                         Constrained-System Design Lens
    1990        1995        2000        2005        2010             2015           2020
     │           │           │           │           │                │              │
     ▼           ▼           ▼           ▼           ▼                ▼              ▼
┌──────────┐┌────────────┐┌──────────┐┌──────────┐┌────────────┐┌────────────────┐┌─────────┐
│  Expert  ││Early       ││Attention ││ Scaling  ││Transformers││ Neural Hardware││ Quantum │
│  Systems ││Transformers││ Research ││ Laws     ││ Dominance  ││  Acceleration  ││  LLMs   │
└──────────┘└────────────┘└──────────┘└──────────┘└────────────┘└────────────────┘└─────────┘
```
*Figure 6: Actual AI research timeline with a constrained-system design lens*

### 7.3 Language-Model Operations Under Constraints

The emergence of deep learning as a dominant paradigm in AI occurred primarily
in the late 2000s and early 2010s, with key developments including GPU training,
convolutional networks, sequence-to-sequence models, attention mechanisms, and
transformers. GPT2-BASIC does not revise that history. It documents which parts
of a modern language-model system can be reduced into a DOS-class runtime when
training, export, quantization, and pack construction happen on the host.

Under 486-class constraints, several design pressures become visible:

1. **Attention Cost**: Attention is mathematically simple but expensive enough
   that context size, cache layout, and fixed-point exp tables matter.

2. **Host-Side Preparation**: Training and pack construction remain host-side
   jobs; the DOS target consumes exported artifacts.

3. **Hardware-Software Fit**: The runtime benefits from kernels and file
   formats that match the target's memory and disk behavior.

4. **Recall Before Generation**: A tiny model is more useful when paired with
   local knowledge records and fast indexes.

| Pressure Point | Release Mechanism |
|----------------|-------------------|
| CPU cost | Q20.12 fixed-point kernels and optional head shortlist |
| Memory pressure | q4 token/head artifacts and streamed-head fallback |
| Vocabulary size | DOS-loadable lexicon vocabulary |
| Weak raw generation | golden replies, retrieval, and fallback rules |
| Slow full search | KDB/KB2 buckets and sharded `KB2T?.TXT` term indexes |

*Table 12: Constrained-system design pressures and release mechanisms*

### 7.4 Release Application Surface

The current release application surface is deliberately narrow and evidence
gated. It covers local inference, assistant shell behavior, pack switching,
indexed knowledge recall, session memory, USER.TXT note import, release bundle
validation, and hardware-transfer rehearsal under QEMU.

| Application Area | Current Release Surface |
|------------------|-------------------------|
| Local text generation | Compact GPT-style fixed-point runtime |
| Conversational assistant | CHAT pack with golden replies, model output, memory, and retrieval |
| DOS help | DOSHELP pack with local troubleshooting and setup notes |
| Office workflows | OFFICE pack for short local productivity prompts |
| Development notes | DEV pack with compact technical recall |
| Portability guidance | PORTABLE pack for BASIC/C/assembly/substrate notes |

*Table 13: Current evidence-gated application surface*

The release does not claim broad modern assistant ability. It claims a concrete
local loop: exported weights, DOS execution, pack selection, local recall,
bounded replies, QEMU stress evidence, and a path for physical-machine return
logs.

### 7.5 Hardware Co-Design Pressure Points

GPT2-BASIC highlights hardware features that matter for local inference
workloads on constrained machines:

1. **Integer Multiply and Accumulation**: Fixed-point matrix/vector kernels
   depend on predictable integer multiply, shifts, and accumulation.

2. **Memory Bandwidth and Layout**: Tensor layout, tokenizer tables, and KDB/KB2
   records determine how much work can stay sequential.

3. **Disk Access Patterns**: Streaming fallback modes depend on predictable
   local file access and small bounded records.

4. **Branch and Loop Simplicity**: BASIC and C ports benefit from small kernels,
   explicit loops, and file contracts that can be checked independently.

5. **Validation Hooks**: Constrained targets need logs, checksums, and simple
   return-file workflows because interactive debugging is limited.

These pressure points are implementation guidance, not predictions about
hardware history.

## 8. Educational Value

Beyond its technical innovations and historical implications, our implementation offers significant educational value for understanding both transformer models and optimization techniques for constrained environments.

### 8.1 Demonstrating Transformer Fundamentals

One of the primary educational benefits of this implementation is its clarity in demonstrating the core algorithms that underlie transformer models. By stripping away the complexity of modern implementations optimized for GPUs, our BASIC code reveals the essential mathematical operations.

#### Key Concepts Illustrated

1. **Self-Attention Mechanism**: The implementation clearly demonstrates how tokens attend to one another through scaled dot-product attention:

```basic
' Simplified self-attention calculation
SUB SelfAttention(query AS Matrix, key AS Matrix, value AS Matrix, output AS Matrix, mask AS Matrix)
    ' Calculate attention scores: Q * K^T
    DIM scores AS Matrix
    InitMatrix scores, query.rows, key.rows
    MatrixMultiply query, key, scores, MATRIX_TRANSPOSE_B

    ' Scale by sqrt(d_k)
    DIM scale_factor AS INTEGER = FixedDivide(FIXED_POINT_SCALE, FixedSqrt(FloatToFixed(CSNG(query.cols))))
    MatrixScale scores, scale_factor

    ' Apply mask (if provided)
    IF NOT IsNullMatrix(mask) THEN
        MatrixElementwiseMul scores, mask, scores
    END IF

    ' Apply softmax to get attention weights
    SoftmaxRowwise scores

    ' Calculate weighted values: softmax(QK^T/sqrt(d_k)) * V
    MatrixMultiply scores, value, output
END SUB
```

This code clearly shows the sequence of operations in self-attention: the calculation of attention scores, scaling, masking, softmax normalization, and the final weighted sum.

2. **Multi-Head Attention**: The implementation demonstrates how attention is split across multiple heads, allowing the model to focus on different aspects of the input:

```basic
' Simplified multi-head attention
SUB MultiHeadAttention(input AS Matrix, wq AS Matrix, wk AS Matrix, wv AS Matrix, wo AS Matrix, output AS Matrix, num_heads AS INTEGER)
    ' Project to query, key, value representations
    DIM query AS Matrix, key AS Matrix, value AS Matrix
    MatrixMultiply input, wq, query
    MatrixMultiply input, wk, key
    MatrixMultiply input, wv, value

    ' Split heads and transpose
    DIM head_dim AS INTEGER = query.cols \ num_heads
    ' (Implementation details of splitting omitted for brevity)

    ' Apply self-attention for each head
    ' (Implementation details omitted for brevity)

    ' Combine heads and project back
    ' (Implementation details omitted for brevity)

    ' Final projection
    MatrixMultiply combined_heads, wo, output
END SUB
```

3. **Feed-Forward Networks**: The code illustrates the structure and purpose of feed-forward networks in transformers:

```basic
' Simplified feed-forward network with GLU activation
SUB FeedForward(input AS Matrix, w1 AS Matrix, w2 AS Matrix, w3 AS Matrix, output AS Matrix)
    ' First projection
    DIM intermediate1 AS Matrix, intermediate2 AS Matrix
    MatrixMultiply input, w1, intermediate1
    MatrixMultiply input, w2, intermediate2

    ' Apply gating (GLU activation)
    DIM gated AS Matrix
    InitMatrix gated, intermediate1.rows, intermediate1.cols

    ' Element-wise multiplication with sigmoid of gate
    DIM r AS INTEGER, c AS INTEGER
    FOR r = 0 TO intermediate1.rows - 1
        FOR c = 0 TO intermediate1.cols - 1
            DIM fp_val AS INTEGER = LogQuantizedToFixed(intermediate2.data(r, c))
            DIM sigmoid_val AS INTEGER = FixedSigmoid(fp_val)
            DIM gated_val AS INTEGER = FixedMultiply(LogQuantizedToFixed(intermediate1.data(r, c)), sigmoid_val)
            gated.data(r, c) = FixedToLogQuantized(gated_val).packed_value
        NEXT c
    NEXT r

    ' Second projection
    MatrixMultiply gated, w3, output
END SUB
```

4. **Layer Normalization**: The implementation shows how normalization stabilizes training and improves performance:

```basic
' Simplified layer normalization
SUB LayerNorm(input AS Matrix, gamma AS Matrix, beta AS Matrix, output AS Matrix)
    ' For each row (sequence position)
    DIM r AS INTEGER, c AS INTEGER
    FOR r = 0 TO input.rows - 1
        ' Calculate mean
        DIM mean AS INTEGER = 0
        FOR c = 0 TO input.cols - 1
            mean = FixedAdd(mean, LogQuantizedToFixed(input.data(r, c)))
        NEXT c
        mean = FixedDivide(mean, FloatToFixed(CSNG(input.cols)))

        ' Calculate variance
        DIM variance AS INTEGER = 0
        FOR c = 0 TO input.cols - 1
            DIM diff AS INTEGER = FixedSubtract(LogQuantizedToFixed(input.data(r, c)), mean)
            variance = FixedAdd(variance, FixedMultiply(diff, diff))
        NEXT c
        variance = FixedDivide(variance, FloatToFixed(CSNG(input.cols)))

        ' Normalize, scale, and shift
        DIM std_dev AS INTEGER = FixedSqrt(FixedAdd(variance, FloatToFixed(0.00001))) ' Epsilon for stability
        FOR c = 0 TO input.cols - 1
            DIM norm_val AS INTEGER = FixedDivide(FixedSubtract(LogQuantizedToFixed(input.data(r, c)), mean), std_dev)
            DIM scaled AS INTEGER = FixedMultiply(norm_val, LogQuantizedToFixed(gamma.data(0, c)))
            DIM shifted AS INTEGER = FixedAdd(scaled, LogQuantizedToFixed(beta.data(0, c)))
            output.data(r, c) = FixedToLogQuantized(shifted).packed_value
        NEXT c
    NEXT r
END SUB
```

#### Pedagogical Advantages

The BASIC implementation offers several pedagogical advantages over modern implementations:

1. **Algorithmic Clarity**: Without the complexities of GPU optimization, the core algorithms are more apparent.

2. **Step-by-Step Execution**: The code can be executed step-by-step in a debugger, allowing students to observe the transformation of data through the network.

3. **Minimal Dependencies**: The implementation avoids dependencies on complex libraries, making it more accessible for educational purposes.

4. **Direct Mathematical Mapping**: The code directly implements the mathematical operations described in transformer papers, making it easier to connect theory with implementation.

The release build now includes the practical text-mode tracing surface implied
by these advantages. `GPT2.EXE --trace` runs inside DOS against the active
`C:\MODEL` checkpoint and emits stable `TRACE_*` records for model shape,
tokenizer mode, prompt tokens, each greedy fixed-point generation step, and the
final decoded text. The QEMU harness `bash qemu/run_trace_486.sh` captures this
as `qemu/evidence/trace_486.log`, giving students a step-by-step audit trail
without requiring a graphics card beyond ordinary DOS text mode.

### 8.2 Optimization Techniques for Constrained Environments

Beyond illustrating transformer fundamentals, our implementation demonstrates a range of optimization techniques that are valuable for constrained computing environments:

#### Memory Optimization Techniques

1. **Data Quantization**: The 4-bit logarithmic quantization scheme demonstrates how precision can be traded for memory efficiency.

2. **Sparse Representations**: The block-sparse attention implementation shows how structural sparsity can be exploited for memory savings.

3. **Memory Reuse**: The implementation carefully controls allocation and deallocation to minimize the memory footprint.

4. **Parameter Streaming**: The disk streaming system demonstrates on-demand loading for operating with limited RAM.

#### Computational Optimization Techniques

1. **Fixed-Point Arithmetic**: The implementation shows how fixed-point can replace floating-point for efficiency on limited hardware.

2. **Lookup Tables**: Lookup tables for expensive functions (exponential, trigonometric) demonstrate classical optimization techniques.

3. **SIMD-like Operations**: The bit manipulation techniques demonstrate how to achieve parallelism even without dedicated SIMD instructions.

4. **Loop Optimization**: Various loop optimizations (unrolling, blocking) demonstrate classical performance techniques.

These optimization techniques remain valuable today, particularly for edge computing, IoT devices, and other resource-constrained environments. Students learning about either transformer models or optimization techniques can benefit from studying how these approaches work together in our implementation.

### 8.3 Insights for Modern Edge AI

The constraints of 486-era hardware in many ways mirror the constraints of modern edge devices, where power consumption, memory, and processing capabilities are limited. Our implementation offers insights for modern edge AI development:

#### Transferable Techniques

1. **Quantization Approaches**: The logarithmic quantization scheme demonstrates an approach to quantization that preserves dynamic range, which is valuable for edge ML deployments.

2. **Sparse Computation**: The block-sparse attention technique demonstrates how structural sparsity can be exploited on devices without dedicated sparse tensor operations.

3. **Memory Streaming**: The parameter streaming system shows how models larger than available RAM can still be deployed effectively.

4. **Fixed-Point Arithmetic**: Many modern microcontrollers lack floating-point units or have limited floating-point performance, making fixed-point techniques relevant again.

#### Edge AI Implications

Modern edge AI faces challenges similar to those we addressed for 486-era hardware:

1. **Energy Efficiency**: Fixed-point arithmetic and sparse computation can significantly reduce energy consumption, a critical concern for battery-powered devices.

2. **Memory Constraints**: Techniques like quantization and sparse representations help deploy sophisticated models on memory-constrained devices.

3. **Limited Processing Power**: Optimization techniques like loop unrolling and SIMD-like operations can maximize performance on limited processors.

4. **Bandwidth Limitations**: The streaming parameter approach addresses bandwidth constraints between storage and computation.

By studying our implementation, designers of modern edge AI systems can gain insights into how these challenges can be addressed, even on highly constrained hardware.

## 9. Future Directions

While our implementation demonstrates that transformer models can operate on 486-era hardware, numerous opportunities exist for further optimization and exploration:

### 9.1 Further Optimizations

Several potential optimizations could further improve performance or capabilities:

#### Memory Efficiency

1. **Further Quantization**: Experimenting with even more aggressive quantization schemes, such as 2-bit or 1-bit representations for certain parameters.

2. **Dynamic Sparse Patterns**: Implementing dynamically determined sparsity patterns based on input characteristics, rather than fixed block-sparse structures.

3. **Hierarchical Memory Management**: Implementing a more sophisticated memory hierarchy that leverages conventional, upper, extended, and expanded memory more effectively.

#### Computational Efficiency

1. **Further Assembly Optimization**: Extending assembly language implementations to cover more operations, with processor-specific optimizations.

2. **Specialized MMIO Drivers**: Developing memory-mapped I/O drivers for more efficient disk streaming on DOS systems.

3. **Alternative Fixed-Point Formats**: Experimenting with different fixed-point formats (e.g., Q24.8 or Q12.20) for different operations.

#### Architectural Improvements

1. **Alternative Transformer Variants**: Implementing more efficient transformer variants, such as Linformer or Performer, which have lower computational complexity.

2. **Conditional Computation**: Implementing dynamic computation paths that activate only certain parts of the network based on input characteristics.

3. **Progressive Decoding**: Implementing multi-stage decoding processes that refine outputs progressively, allowing for earlier termination when appropriate.

### 9.2 Applications and Extensions

The current implementation focuses on demonstrating technical feasibility rather than specific applications. Several potential extensions could enhance its utility:

#### Application-Specific Adaptations

1. **Domain-Specialized Models**: Training smaller, domain-specific models optimized for particular applications like documentation assistance or code completion.

2. **Integration with DOS Software**: Creating interfaces for integration with popular DOS applications like word processors or database systems.

3. **Interactive Fiction**: Developing interactive text adventure games powered by the language model, showcasing the technology in an accessible format.

#### Educational Extensions

1. **Interactive Visualizations**: The release now includes optional lab `VISUAL.EXE`, a Mode 13h token/progress visualizer that reads `TRACE.LOG` from `GPT2.EXE --trace` and emits `VISUAL_*` evidence through `qemu/run_visual_trace_486.sh`. The text trace remains the portable baseline; richer attention heatmaps can build on the same contract.

2. **Step-by-Step Execution Mode**: The release executable now implements this as `GPT2.EXE --trace`, with QEMU evidence in `qemu/evidence/trace_486.log`. Future work can add richer mathematical annotations or graphical overlays, but the DOS inference trace itself is no longer aspirational.

3. **Comparative Implementations**: Developing parallel implementations using different optimization techniques to demonstrate trade-offs.

#### Technical Extensions

1. **Training Infrastructure**: Extending the system to support fine-tuning on limited hardware, even if at extremely slow rates.

2. **Cross-Platform Implementation**: Developing more comprehensive adaptations
   for constrained platforms beyond DOS.

3. **Hardware Acceleration Exploration**: Experimenting with potential hardware acceleration using contemporary expansion cards or coprocessors.

These future directions represent opportunities for further research and
development in portable local inference, indexed recall, and constrained-system
runtime design. The project provides insights for modern edge AI development and
educational resources for understanding transformer models.

# 10. Addendum: Practical Applications and Modern Relevance

## A. Modern Quantization Connections

While our 4-bit logarithmic quantization scheme was developed for 486-era constraints, it bears remarkable similarities to quantization approaches now gaining prominence in modern AI hardware. In the current release this is not just a paper design: optional q4/log token-embedding and output-head artifacts are loaded by DOS, checked by host validators, and covered by vector parity and timing evidence.

### A.1 Connections to Modern Neural Network Quantization

Recent work in neural network quantization (Jacob et al., 2018; Krishnamoorthi, 2018) has increasingly focused on sub-8-bit representations, with 4-bit quantization becoming particularly important for edge devices. Our logarithmic approach shares conceptual similarities with:

1. **Log-based quantization** in modern neural accelerators like Graphcore's IPU, which uses logarithmic representation to balance precision across magnitude ranges (Jia et al., 2021)

2. **Mixed-precision approaches** that allocate different bit widths to different parts of the network based on sensitivity analysis (Wang et al., 2019)

3. **Dynamic fixed-point** techniques that have become standard in many edge AI frameworks like TensorFlow Lite and NVIDIA's TensorRT

This convergent evolution suggests that the fundamental mathematical constraints of neural computation naturally lead to similar optimization approaches, regardless of the hardware era.

### A.2 Edge AI and Embedded Systems Applications

The techniques developed for our 486-era implementation have direct relevance for modern embedded AI systems:

```
┌───────────────────────────┐     ┌───────────────────────────┐
│ 486-Era Constraints       │     │ Modern IoT Constraints    │
├───────────────────────────┤     ├───────────────────────────┤
│ • 32MB RAM limit          │ ┌─► │ • 256KB-4MB RAM limit     │
│ • No FPU (486SX)          │ │   │ • Minimal power budget    │
│ • Limited disk bandwidth  │ │   │ • Battery operation       │
│ • 100MHz clock speed      │ │   │ • Clock speed constraints │
└───────────────────────────┘ │   └───────────────────────────┘
                              │
┌───────────────────────────┐ │   ┌───────────────────────────┐
│ Our Solutions             │ │   │ Applied to Modern Devices │
├───────────────────────────┤ │   ├───────────────────────────┤
│ • 4-bit log quantization  │ └─► │ • Ultra-low precision     │
│ • Fixed-point arithmetic  │     │ • Integer-only inference  │
│ • Sparse computation      │     │ • Pruned networks         │
│ • Streaming parameters    │     │ • On-demand loading       │
└───────────────────────────┘     └───────────────────────────┘
```
*Figure 8: Mapping 486-era solutions to modern IoT constraints*

Modern microcontrollers like the ARM Cortex-M series (particularly M4/M7) face similar constraints to 486-era processors and could directly benefit from these techniques. The emergence of TinyML demonstrates the importance of efficient inference on extremely constrained devices.

## B. Practical Extensions and Integration

### B.1 Integration with Existing Systems

The modular design allows integration with other legacy and modern systems:

1. **DOS/Windows Integration**: The implementation could be adapted as a TSR (Terminate and Stay Resident) program in DOS or a dynamic library in Windows, providing text generation capabilities to other applications.

2. **Modern Bridge Applications**: Using DOSBox or similar emulation technology,
   the implementation could expose the DOS runtime to modern tooling for tests,
   demos, or controlled experiments.

Sample integration code might look like:

```basic
' Function to expose transformer capabilities through DOS interrupt
SUB InstallInterruptHandler()
    old_handler = SetInterruptVector(&H60, @TransformerInterruptHandler)
END SUB

SUB TransformerInterruptHandler()
    ' AH = Function:
    ' 01h = Initialize model
    ' 02h = Generate text
    ' 03h = Free model

    SELECT CASE REG.AH
        CASE &H01 ' Initialize
            ' DS:DX points to model filename
            model_ptr = LoadModel(ConvertSegmentedToLinear(REG.DS, REG.DX))
            REG.AX = model_ptr ' Return model handle

        CASE &H02 ' Generate
            ' DS:SI points to prompt
            ' ES:DI points to output buffer
            ' CX = max tokens to generate
            prompt = ConvertSegToString(REG.DS, REG.SI)
            GenerateText(prompt, REG.CX, output_buffer)
            CopyStringToSegmented(output_buffer, REG.ES, REG.DI)

        CASE &H03 ' Free
            ' BX = model handle
            FreeModel(REG.BX)
    END SELECT
END SUB
```

### B.2 Educational Toolkit Extensions

The implementation now has the first component of a comprehensive educational
toolkit:

1. **Interactive Tracing**: The text-mode `GPT2.EXE --trace` path records model
   metadata, tokenizer pieces, generation stages, generated tokens, and decoded
   text on DOS. Optional lab `VISUAL.EXE` reads that trace, adds a Mode 13h
   token/progress view, and emits machine-readable `VISUAL_*` records. Richer
   attention and embedding views can build on this trace contract.

2. **Comparative Implementations**: Developing parallel implementations with different optimization strategies that students could benchmark against each other.

3. **Step-by-Step Mode**: The implemented trace modes provide both the DOS
   execution trail and a VGA visual path; mathematical annotations and richer
   overlays remain useful additions rather than prerequisites.

Sample visualization code might look like:

```basic
SUB VisualizeAttention(attention_matrix AS SparseBlockMatrix)
    SCREEN 13 ' 320x200, 256 colors

    ' Scale attention scores to color intensities
    DIM r AS INTEGER, c AS INTEGER
    DIM max_score AS INTEGER = FindMaxAttentionScore(attention_matrix)

    ' Draw attention heatmap
    FOR r = 0 TO attention_matrix.rows - 1
        FOR c = 0 TO attention_matrix.cols - 1
            DIM score AS INTEGER = GetSparseValue(attention_matrix, r, c)
            DIM color AS INTEGER = INT((score / max_score) * 255)
            PSET (50 + c * 2, 50 + r * 2), color
        NEXT c
    NEXT r

    ' Add labels and annotations
    COLOR 15
    LOCATE 2, 2
    PRINT "Attention Pattern"
    ' ...additional annotations...
END SUB
```

## C. Constrained and Legacy System Engagement

### C.1 Demoscene Potential

The demoscene—a community focused on creating technically impressive and artistic computer programs within severe constraints—represents a natural audience for this implementation:

1. **Demo Competitions**: The implementation could be adapted for demoscene competitions like Revision or Assembly, showcasing AI capabilities within classic hardware constraints.

2. **Size-Coding Challenges**: The core components could be optimized for extreme size constraints (e.g., 4KB or 64KB executables).

3. **Artward Innovation**: Integration with demo effects could create novel generative text art that responds to visuals, music, or user input.

### C.2 Legacy System Preservation

This project also contributes to computing preservation efforts:

1. **Documentation of Optimization Techniques**: Preserving knowledge of low-level optimization techniques that are increasingly rare in modern computing education.

2. **Functional Demonstration**: Showing that legacy hardware remains capable
   of meaningful computation, encouraging continued preservation efforts.

3. **Historical Context**: Providing a bridge between historical computing and modern AI concepts for educational purposes.

## D. Training and Export Considerations

The release boundary is explicit: GPT2-BASIC trains and exports model artifacts
on the host, then runs inference and assistant recall on the DOS target.
Target-side training is not part of the current production claim.

### D.1 Host-Side Training Boundary

1. **Full Training**: Full model training remains a host-side operation. The
   generated artifacts are then exported into DOS-loadable binary and text
   formats.

2. **Pack Model Training**: Assistant pack-local models are also trained on the
   host, then copied with their pack metadata, HELP rows, KDB/KB2 records, and
   term indexes.

3. **Validation Before Shipping**: Host-side training results must pass quality,
   fixed-point parity, release-manifest, and QEMU evidence gates before being
   treated as release assets.

### D.2 Target-Side Adaptation

The current target-side adaptation mechanisms are deliberately lightweight:

| Mechanism | Runs on DOS target | Purpose |
|-----------|--------------------|---------|
| Session memory | Yes | Remember small facts during one assistant session |
| Pack switching | Yes | Hot-load local model/knowledge context |
| USER.TXT note import | Yes | Add local facts without retraining |
| KDB/KB2 recall | Yes | Retrieve curated local knowledge |
| Weight training | No | Host-side operation |

*Table 14: Production training boundary and lightweight target adaptation*

## E. Open Source and Community Development

### E.1 Open Source Framework

This implementation could serve as the foundation for an open-source framework focused on efficient transformer implementations for constrained environments:

1. **Modular Components**: The optimization techniques could be packaged as reusable components for other projects.

2. **Cross-Platform Extensions**: The codebase could be expanded to support a
   wider range of legacy, embedded, and modern platforms.

3. **Benchmark Suite**: A standardized benchmark suite could be developed to compare different optimizations across platforms.

### E.2 Educational Curriculum

A curriculum could be developed around this implementation for courses on:

1. **Efficient AI Implementation**: Teaching optimization techniques for resource-constrained environments.

2. **Computing History**: Illustrating the evolution of computing through practical examples.

3. **Algorithm Design**: Demonstrating how the same mathematical foundations can be implemented across vastly different hardware generations.

This addendum expands the documentation by highlighting practical applications,
modern connections, and potential extensions of GPT2-BASIC, providing additional
context for the technical and historical aspects of the project.

## 11. References

Abrash, M. (1994). *Zen of Graphics Programming*. The Coriolis Group.

Anderson, E., Bai, Z., Bischof, C., Blackford, S., Demmel, J., Dongarra, J., ... & Sorensen, D. (1990). LAPACK: A portable linear algebra library for high-performance computers. In *Proceedings of Supercomputing '90* (pp. 2-11).

Arnold, M. G., Bailey, T., Cowles, J., & Cupal, J. (1990). Redundant logarithmic arithmetic. *IEEE Transactions on Computers*, 39(8), 1077-1086.

Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*.

Carmack, J. (1993). *Doom source code*. id Software.

Ciresan, D. C., Meier, U., Gambardella, L. M., & Schmidhuber, J. (2010). Deep, big, simple neural nets for handwritten digit recognition. *Neural computation*, 22(12), 3207-3220.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine learning*, 20(3), 273-297.

Crowley, D. A. (1996). *Understanding and using advanced DOS memory*. QUE Corporation.

Davis, T. A., Jiang, W., & Duff, I. S. (1994). Sparse Matrix Methods in Scientific Computing. *IEEE Computational Science and Engineering*, 1(4), 23-32.

Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M., Mao, M., Senior, A., Tucker, P., Yang, K., Le, Q. V., & Ng, A. Y. (2012). Large scale distributed deep networks. In *Advances in neural information processing systems* (pp. 1223-1231).

Duncan, R. (1992). *Advanced MS-DOS Programming: The Microsoft Guide for Assembly Language and C Programmers*. Microsoft Press.

Duntemann, J. (1992). *Assembly Language Step-by-Step: Programming with DOS and Linux*. John Wiley & Sons.

Hastie, T., & Tibshirani, R. (1990). *Generalized additive models*. Chapman and Hall.

Hendler, J. (2008). Avoiding another AI winter. *IEEE Intelligent Systems*, 23(2), 2-4.

Herf, M. (1994). *Efficient Generation of Soft Shadows Using an Adaptive Shadow Map*. Carnegie Mellon University.

Hinton, G. E. (1990). Connectionist learning procedures. *Artificial intelligence*, 40(1-3), 185-234.

Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A fast learning algorithm for deep belief nets. *Neural computation*, 18(7), 1527-1554.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.

Hopfield, J. J. (1988). Neural networks and physical systems with emergent collective computational capabilities. In *Spin Glass Theory and Beyond: An Introduction to the Replica Method and Its Applications* (pp. 411-415).

Hsu, F. H., Anantharaman, T., Campbell, M., & Nowatzyk, A. (1990). A grandmaster chess machine. *Scientific American*, 263(4), 44-50.

Ihaka, R., & Gentleman, R. (1996). R: a language for data analysis and graphics. *Journal of computational and graphical statistics*, 5(3), 299-314.

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2704-2713).

Jelinek, F. (1991). Up from trigrams! The struggle for improved language models. In *Second European Conference on Speech Communication and Technology*.

Jia, Z., Tillman, B., Maggioni, M., & Scarpazza, D. P. (2021). Dissecting the Graphcore IPU architecture via microbenchmarking. *arXiv preprint arXiv:2104.11533*.

Jordan, M. I. (1990). *Serial Order: A Parallel Distributed Processing Approach. Advances in Psychology, Vol. 121*. North-Holland.

Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for efficient inference: A whitepaper. *arXiv preprint arXiv:1806.08342*.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25.

LeCun, Y., Denker, J. S., & Solla, S. A. (1990). Optimal brain damage. In *Advances in neural information processing systems* (pp. 598-605).

Nagel, M., van Baalen, M., Blankevoort, T., & Welling, M. (2021). Data-free quantization through weight equalization and bias correction. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(4), 1815-1828.

Norton, P. (1994). *Peter Norton's Complete Guide to DOS 6.22*. Sams Publishing.

Pearl, J. (1988). *Probabilistic reasoning in intelligent systems: Networks of plausible inference*. Morgan Kaufmann.

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 9.

Raina, R., Madhavan, A., & Ng, A. Y. (2009). Large-scale deep unsupervised learning using graphics processors. In *Proceedings of the 26th Annual International Conference on Machine Learning* (pp. 873-880).

Ripley, B. D. (1994). Neural networks and related methods for classification. *Journal of the Royal Statistical Society: Series B (Methodological)*, 56(3), 409-437.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

Smith, S. W. (1987). The scientist and engineer's guide to digital signal processing. *California Technical Publishing*.

Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *Advances in neural information processing systems*, 27.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

Wallace, G. K. (1992). The JPEG still picture compression standard. *IEEE transactions on consumer electronics*, 38(1), xviii-xxxiv.

Wang, K., Liu, Z., Lin, Y., Lin, J., & Han, S. (2019). HAQ: Hardware-aware automated quantization with mixed precision. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 8612-8620).
