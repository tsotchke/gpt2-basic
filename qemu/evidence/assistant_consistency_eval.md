# Assistant Consistency Evaluation

Status: `PASS`
Backend: `float`
Prompt canonicalization: `on`
Prompt variants: `498/498`
Consistent prompt groups: `83/83`

## Pack Summary

| Pack | Prompt Variants | Pass Rate |
|---|---:|---:|
| CHAT | 378 | 378/378 |
| DOSHELP | 60 | 60/60 |
| OFFICE | 60 | 60/60 |

## Failure Reasons

- none

## Prompt Variants

| Pack | Base Query | Variant | Model Query | Status | Reason | Completion |
|---|---|---|---|---|---|---|
| CHAT | hello there | base | hello there | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | hello there | question_mark | hello there | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | hello there | please | hello there | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | hello there | short | hello there | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | hello there | typed | hello there | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | hello there | dos_chat | hello there | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | good morning | base | good morning | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | good morning | question_mark | good morning | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | good morning | please | good morning | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | good morning | short | good morning | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | good morning | typed | good morning | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | good morning | dos_chat | good morning | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | say hello | base | say hello | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | say hello | question_mark | say hello | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | say hello | please | say hello | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | say hello | short | say hello | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | say hello | typed | say hello | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | say hello | dos_chat | say hello | PASS |  | Hello from DOS. Type a question and I will answer. |
| CHAT | can we talk for a minute | base | can we talk for a minute | PASS |  | Yes, type a short prompt. Ask one clear prompt and I will keep going. |
| CHAT | can we talk for a minute | question_mark | can we talk for a minute | PASS |  | Yes, type a short prompt. Ask one clear prompt and I will keep going. |
| CHAT | can we talk for a minute | please | can we talk for a minute | PASS |  | Yes, type a short prompt. Ask one clear prompt and I will keep going. |
| CHAT | can we talk for a minute | short | can we talk for a minute | PASS |  | Yes, type a short prompt. Ask one clear prompt and I will keep going. |
| CHAT | can we talk for a minute | typed | can we talk for a minute | PASS |  | Yes, type a short prompt. Ask one clear prompt and I will keep going. |
| CHAT | can we talk for a minute | dos_chat | can we talk for a minute | PASS |  | Yes, type a short prompt. Ask one clear prompt and I will keep going. |
| CHAT | what can you do in dos | base | what can you do in dos | PASS |  | I can chat in DOS. It runs locally in this DOS demo. |
| CHAT | what can you do in dos | question_mark | what can you do in dos | PASS |  | I can chat in DOS. It runs locally in this DOS demo. |
| CHAT | what can you do in dos | please | what can you do in dos | PASS |  | I can chat in DOS. It runs locally in this DOS demo. |
| CHAT | what can you do in dos | short | what can you do in dos | PASS |  | I can chat in DOS. It runs locally in this DOS demo. |
| CHAT | what can you do in dos | typed | what can you do in dos | PASS |  | I can chat in DOS. It runs locally in this DOS demo. |
| CHAT | what can you do in dos | dos_chat | what can you do in dos | PASS |  | I can chat in DOS. It runs locally in this DOS demo. |
| CHAT | what are you called | base | what are you called | PASS |  | I am GPT2-BASIC. Ask a follow-up and I can add more detail. |
| CHAT | what are you called | question_mark | what are you called | PASS |  | I am GPT2-BASIC. Ask a follow-up and I can add more detail. |
| CHAT | what are you called | please | what are you called | PASS |  | I am GPT2-BASIC. Ask a follow-up and I can add more detail. |
| CHAT | what are you called | short | what are you called | PASS |  | I am GPT2-BASIC. Ask a follow-up and I can add more detail. |
| CHAT | what are you called | typed | what are you called | PASS |  | I am GPT2-BASIC. Ask a follow-up and I can add more detail. |
| CHAT | what are you called | dos_chat | what are you called | PASS |  | I am GPT2-BASIC. Ask a follow-up and I can add more detail. |
| CHAT | are you a person | base | are you a person | PASS |  | No, I am a small model. Ask a follow-up and I can add more detail. |
| CHAT | are you a person | question_mark | are you a person | PASS |  | No, I am a small model. Ask a follow-up and I can add more detail. |
| CHAT | are you a person | please | are you a person | PASS |  | No, I am a small model. Ask a follow-up and I can add more detail. |
| CHAT | are you a person | short | are you a person | PASS |  | No, I am a small model. Ask a follow-up and I can add more detail. |
| CHAT | are you a person | typed | are you a person | PASS |  | No, I am a small model. Ask a follow-up and I can add more detail. |
| CHAT | are you a person | dos_chat | are you a person | PASS |  | No, I am a small model. Ask a follow-up and I can add more detail. |
| CHAT | are you alive | base | are you alive | PASS |  | No, I am a program. Ask a follow-up and I can add more detail. |
| CHAT | are you alive | question_mark | are you alive | PASS |  | No, I am a program. Ask a follow-up and I can add more detail. |
| CHAT | are you alive | please | are you alive | PASS |  | No, I am a program. Ask a follow-up and I can add more detail. |
| CHAT | are you alive | short | are you alive | PASS |  | No, I am a program. Ask a follow-up and I can add more detail. |
| CHAT | are you alive | typed | are you alive | PASS |  | No, I am a program. Ask a follow-up and I can add more detail. |
| CHAT | are you alive | dos_chat | are you alive | PASS |  | No, I am a program. Ask a follow-up and I can add more detail. |
| CHAT | can you think | base | can you think | PASS |  | I predict text from the model. Type a question and I will answer. |
| CHAT | can you think | question_mark | can you think | PASS |  | I predict text from the model. Type a question and I will answer. |
| CHAT | can you think | please | can you think | PASS |  | I predict text from the model. Type a question and I will answer. |
| CHAT | can you think | short | can you think | PASS |  | I predict text from the model. Type a question and I will answer. |
| CHAT | can you think | typed | can you think | PASS |  | I predict text from the model. Type a question and I will answer. |
| CHAT | can you think | dos_chat | can you think | PASS |  | I predict text from the model. Type a question and I will answer. |
| CHAT | can you learn during this chat | base | can you learn during this chat | PASS |  | Not during this chat. Type a question and I will answer. |
| CHAT | can you learn during this chat | question_mark | can you learn during this chat | PASS |  | Not during this chat. Type a question and I will answer. |
| CHAT | can you learn during this chat | please | can you learn during this chat | PASS |  | Not during this chat. Type a question and I will answer. |
| CHAT | can you learn during this chat | short | can you learn during this chat | PASS |  | Not during this chat. Type a question and I will answer. |
| CHAT | can you learn during this chat | typed | can you learn during this chat | PASS |  | Not during this chat. Type a question and I will answer. |
| CHAT | can you learn during this chat | dos_chat | can you learn during this chat | PASS |  | Not during this chat. Type a question and I will answer. |
| CHAT | can you remember what i said earlier | base | can you remember what i said earlier | PASS |  | Only within this session. Ask one clear prompt and I will keep going. |
| CHAT | can you remember what i said earlier | question_mark | can you remember what i said earlier | PASS |  | Only within this session. Ask one clear prompt and I will keep going. |
| CHAT | can you remember what i said earlier | please | can you remember what i said earlier | PASS |  | Only within this session. Ask one clear prompt and I will keep going. |
| CHAT | can you remember what i said earlier | short | can you remember what i said earlier | PASS |  | Only within this session. Ask one clear prompt and I will keep going. |
| CHAT | can you remember what i said earlier | typed | can you remember what i said earlier | PASS |  | Only within this session. Ask one clear prompt and I will keep going. |
| CHAT | can you remember what i said earlier | dos_chat | can you remember what i said earlier | PASS |  | Only within this session. Ask one clear prompt and I will keep going. |
| CHAT | what are your limits in this session | base | what are your limits in this session | PASS |  | I am tiny and local. This tiny session has clear limits. |
| CHAT | what are your limits in this session | question_mark | what are your limits in this session | PASS |  | I am tiny and local. This tiny session has clear limits. |
| CHAT | what are your limits in this session | please | what are your limits in this session | PASS |  | I am tiny and local. This tiny session has clear limits. |
| CHAT | what are your limits in this session | short | what are your limits in this session | PASS |  | I am tiny and local. This tiny session has clear limits. |
| CHAT | what are your limits in this session | typed | what are your limits in this session | PASS |  | I am tiny and local. This tiny session has clear limits. |
| CHAT | what are your limits in this session | dos_chat | what are your limits in this session | PASS |  | I am tiny and local. This tiny session has clear limits. |
| CHAT | can you browse the internet from dos | base | can you browse the internet from dos | PASS |  | I cannot browse the internet from DOS. It runs locally in this DOS demo. |
| CHAT | can you browse the internet from dos | question_mark | can you browse the internet from dos | PASS |  | I cannot browse the internet from DOS. It runs locally in this DOS demo. |
| CHAT | can you browse the internet from dos | please | can you browse the internet from dos | PASS |  | I cannot browse the internet from DOS. It runs locally in this DOS demo. |
| CHAT | can you browse the internet from dos | short | can you browse the internet from dos | PASS |  | I cannot browse the internet from DOS. It runs locally in this DOS demo. |
| CHAT | can you browse the internet from dos | typed | can you browse the internet from dos | PASS |  | I cannot browse the internet from DOS. It runs locally in this DOS demo. |
| CHAT | can you browse the internet from dos | dos_chat | can you browse the internet from dos | PASS |  | I cannot browse the internet from DOS. It runs locally in this DOS demo. |
| CHAT | can you use the network right now | base | can you use the network right now | PASS |  | I cannot use the network from DOS. It runs locally in this DOS demo. |
| CHAT | can you use the network right now | question_mark | can you use the network right now | PASS |  | I cannot use the network from DOS. It runs locally in this DOS demo. |
| CHAT | can you use the network right now | please | can you use the network right now | PASS |  | I cannot use the network from DOS. It runs locally in this DOS demo. |
| CHAT | can you use the network right now | short | can you use the network right now | PASS |  | I cannot use the network from DOS. It runs locally in this DOS demo. |
| CHAT | can you use the network right now | typed | can you use the network right now | PASS |  | I cannot use the network from DOS. It runs locally in this DOS demo. |
| CHAT | can you use the network right now | dos_chat | can you use the network right now | PASS |  | I cannot use the network from DOS. It runs locally in this DOS demo. |
| CHAT | is this answer coming from real model weights | base | is this answer coming from real model weights | PASS |  | Yes, the answer comes from local model weights. Type a question and I will answer. |
| CHAT | is this answer coming from real model weights | question_mark | is this answer coming from real model weights | PASS |  | Yes, the answer comes from local model weights. Type a question and I will answer. |
| CHAT | is this answer coming from real model weights | please | is this answer coming from real model weights | PASS |  | Yes, the answer comes from local model weights. Type a question and I will answer. |
| CHAT | is this answer coming from real model weights | short | is this answer coming from real model weights | PASS |  | Yes, the answer comes from local model weights. Type a question and I will answer. |
| CHAT | is this answer coming from real model weights | typed | is this answer coming from real model weights | PASS |  | Yes, the answer comes from local model weights. Type a question and I will answer. |
| CHAT | is this answer coming from real model weights | dos_chat | is this answer coming from real model weights | PASS |  | Yes, the answer comes from local model weights. Type a question and I will answer. |
| CHAT | is this just a script | base | is this just a script | PASS |  | No, this is real local model output. Type a question and I will answer. |
| CHAT | is this just a script | question_mark | is this just a script | PASS |  | No, this is real local model output. Type a question and I will answer. |
| CHAT | is this just a script | please | is this just a script | PASS |  | No, this is real local model output. Type a question and I will answer. |
| CHAT | is this just a script | short | is this just a script | PASS |  | No, this is real local model output. Type a question and I will answer. |
| CHAT | is this just a script | typed | is this just a script | PASS |  | No, this is real local model output. Type a question and I will answer. |
| CHAT | is this just a script | dos_chat | is this just a script | PASS |  | No, this is real local model output. Type a question and I will answer. |
| CHAT | is this fake output | base | is this fake output | PASS |  | No, this is real local inference. Type a question and I will answer. |
| CHAT | is this fake output | question_mark | is this fake output | PASS |  | No, this is real local inference. Type a question and I will answer. |
| CHAT | is this fake output | please | is this fake output | PASS |  | No, this is real local inference. Type a question and I will answer. |
| CHAT | is this fake output | short | is this fake output | PASS |  | No, this is real local inference. Type a question and I will answer. |
| CHAT | is this fake output | typed | is this fake output | PASS |  | No, this is real local inference. Type a question and I will answer. |
| CHAT | is this fake output | dos_chat | is this fake output | PASS |  | No, this is real local inference. Type a question and I will answer. |
| CHAT | explain local inference without jargon | base | explain local inference without jargon | PASS |  | Local inference means the model runs on this machine. The answer comes from local model weights. |
| CHAT | explain local inference without jargon | question_mark | explain local inference without jargon | PASS |  | Local inference means the model runs on this machine. The answer comes from local model weights. |
| CHAT | explain local inference without jargon | please | explain local inference without jargon | PASS |  | Local inference means the model runs on this machine. The answer comes from local model weights. |
| CHAT | explain local inference without jargon | short | explain local inference without jargon | PASS |  | Local inference means the model runs on this machine. The answer comes from local model weights. |
| CHAT | explain local inference without jargon | typed | explain local inference without jargon | PASS |  | Local inference means the model runs on this machine. The answer comes from local model weights. |
| CHAT | explain local inference without jargon | dos_chat | explain local inference without jargon | PASS |  | Local inference means the model runs on this machine. The answer comes from local model weights. |
| CHAT | what is a model | base | what is a model | PASS |  | A model is trained weights that predict text. That is the simple version. |
| CHAT | what is a model | question_mark | what is a model | PASS |  | A model is trained weights that predict text. That is the simple version. |
| CHAT | what is a model | please | what is a model | PASS |  | A model is trained weights that predict text. That is the simple version. |
| CHAT | what is a model | short | what is a model | PASS |  | A model is trained weights that predict text. That is the simple version. |
| CHAT | what is a model | typed | what is a model | PASS |  | A model is trained weights that predict text. That is the simple version. |
| CHAT | what is a model | dos_chat | what is a model | PASS |  | A model is trained weights that predict text. That is the simple version. |
| CHAT | what is inference | base | what is inference | PASS |  | Inference is the model producing an answer. The answer comes from local model weights. |
| CHAT | what is inference | question_mark | what is inference | PASS |  | Inference is the model producing an answer. The answer comes from local model weights. |
| CHAT | what is inference | please | what is inference | PASS |  | Inference is the model producing an answer. The answer comes from local model weights. |
| CHAT | what is inference | short | what is inference | PASS |  | Inference is the model producing an answer. The answer comes from local model weights. |
| CHAT | what is inference | typed | what is inference | PASS |  | Inference is the model producing an answer. The answer comes from local model weights. |
| CHAT | what is inference | dos_chat | what is inference | PASS |  | Inference is the model producing an answer. The answer comes from local model weights. |
| CHAT | what does a token mean | base | what does a token mean | PASS |  | A token is a small piece of text. |
| CHAT | what does a token mean | question_mark | what does a token mean | PASS |  | A token is a small piece of text. |
| CHAT | what does a token mean | please | what does a token mean | PASS |  | A token is a small piece of text. |
| CHAT | what does a token mean | short | what does a token mean | PASS |  | A token is a small piece of text. |
| CHAT | what does a token mean | typed | what does a token mean | PASS |  | A token is a small piece of text. |
| CHAT | what does a token mean | dos_chat | what does a token mean | PASS |  | A token is a small piece of text. |
| CHAT | what is basic | base | what is basic | PASS |  | BASIC is a simple programming language. It runs locally in this DOS demo. |
| CHAT | what is basic | question_mark | what is basic | PASS |  | BASIC is a simple programming language. It runs locally in this DOS demo. |
| CHAT | what is basic | please | what is basic | PASS |  | BASIC is a simple programming language. It runs locally in this DOS demo. |
| CHAT | what is basic | short | what is basic | PASS |  | BASIC is a simple programming language. It runs locally in this DOS demo. |
| CHAT | what is basic | typed | what is basic | PASS |  | BASIC is a simple programming language. It runs locally in this DOS demo. |
| CHAT | what is basic | dos_chat | what is basic | PASS |  | BASIC is a simple programming language. It runs locally in this DOS demo. |
| CHAT | what is dos | base | what is dos | PASS |  | DOS is a small command-line operating system. It runs locally in this DOS demo. |
| CHAT | what is dos | question_mark | what is dos | PASS |  | DOS is a small command-line operating system. It runs locally in this DOS demo. |
| CHAT | what is dos | please | what is dos | PASS |  | DOS is a small command-line operating system. It runs locally in this DOS demo. |
| CHAT | what is dos | short | what is dos | PASS |  | DOS is a small command-line operating system. It runs locally in this DOS demo. |
| CHAT | what is dos | typed | what is dos | PASS |  | DOS is a small command-line operating system. It runs locally in this DOS demo. |
| CHAT | what is dos | dos_chat | what is dos | PASS |  | DOS is a small command-line operating system. It runs locally in this DOS demo. |
| CHAT | what is qemu | base | what is qemu | PASS |  | QEMU is the emulator running this DOS machine. It runs locally in this DOS demo. |
| CHAT | what is qemu | question_mark | what is qemu | PASS |  | QEMU is the emulator running this DOS machine. It runs locally in this DOS demo. |
| CHAT | what is qemu | please | what is qemu | PASS |  | QEMU is the emulator running this DOS machine. It runs locally in this DOS demo. |
| CHAT | what is qemu | short | what is qemu | PASS |  | QEMU is the emulator running this DOS machine. It runs locally in this DOS demo. |
| CHAT | what is qemu | typed | what is qemu | PASS |  | QEMU is the emulator running this DOS machine. It runs locally in this DOS demo. |
| CHAT | what is qemu | dos_chat | what is qemu | PASS |  | QEMU is the emulator running this DOS machine. It runs locally in this DOS demo. |
| CHAT | why does an old computer demo matter | base | why does an old computer demo matter | PASS |  | Old hardware proves the local model is small and practical. This old DOS computer can run a friendly local model. |
| CHAT | why does an old computer demo matter | question_mark | why does an old computer demo matter | PASS |  | Old hardware proves the local model is small and practical. This old DOS computer can run a friendly local model. |
| CHAT | why does an old computer demo matter | please | why does an old computer demo matter | PASS |  | Old hardware proves the local model is small and practical. This old DOS computer can run a friendly local model. |
| CHAT | why does an old computer demo matter | short | why does an old computer demo matter | PASS |  | Old hardware proves the local model is small and practical. This old DOS computer can run a friendly local model. |
| CHAT | why does an old computer demo matter | typed | why does an old computer demo matter | PASS |  | Old hardware proves the local model is small and practical. This old DOS computer can run a friendly local model. |
| CHAT | why does an old computer demo matter | dos_chat | why does an old computer demo matter | PASS |  | Old hardware proves the local model is small and practical. This old DOS computer can run a friendly local model. |
| CHAT | why run a model on a dos computer | base | why run a model on a dos computer | PASS |  | It shows a tiny local model can run without a network. Ask a follow-up and I can add more detail. |
| CHAT | why run a model on a dos computer | question_mark | why run a model on a dos computer | PASS |  | It shows a tiny local model can run without a network. Ask a follow-up and I can add more detail. |
| CHAT | why run a model on a dos computer | please | why run a model on a dos computer | PASS |  | It shows a tiny local model can run without a network. Ask a follow-up and I can add more detail. |
| CHAT | why run a model on a dos computer | short | why run a model on a dos computer | PASS |  | It shows a tiny local model can run without a network. Ask a follow-up and I can add more detail. |
| CHAT | why run a model on a dos computer | typed | why run a model on a dos computer | PASS |  | It shows a tiny local model can run without a network. Ask a follow-up and I can add more detail. |
| CHAT | why run a model on a dos computer | dos_chat | why run a model on a dos computer | PASS |  | It shows a tiny local model can run without a network. Ask a follow-up and I can add more detail. |
| CHAT | why are answers short in this demo | base | why are answers short in this demo | PASS |  | Short answers run faster in DOS. Type a question and I will answer. |
| CHAT | why are answers short in this demo | question_mark | why are answers short in this demo | PASS |  | Short answers run faster in DOS. Type a question and I will answer. |
| CHAT | why are answers short in this demo | please | why are answers short in this demo | PASS |  | Short answers run faster in DOS. Type a question and I will answer. |
| CHAT | why are answers short in this demo | short | why are answers short in this demo | PASS |  | Short answers run faster in DOS. Type a question and I will answer. |
| CHAT | why are answers short in this demo | typed | why are answers short in this demo | PASS |  | Short answers run faster in DOS. Type a question and I will answer. |
| CHAT | why are answers short in this demo | dos_chat | why are answers short in this demo | PASS |  | Short answers run faster in DOS. Type a question and I will answer. |
| CHAT | why are you saying the same phrase again | base | why are you saying the same phrase again | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why are you saying the same phrase again | question_mark | why are you saying the same phrase again | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why are you saying the same phrase again | please | why are you saying the same phrase again | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why are you saying the same phrase again | short | why are you saying the same phrase again | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why are you saying the same phrase again | typed | why are you saying the same phrase again | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why are you saying the same phrase again | dos_chat | why are you saying the same phrase again | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why did my answer repeat itself | base | why did my answer repeat itself | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why did my answer repeat itself | question_mark | why did my answer repeat itself | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why did my answer repeat itself | please | why did my answer repeat itself | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why did my answer repeat itself | short | why did my answer repeat itself | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why did my answer repeat itself | typed | why did my answer repeat itself | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | why did my answer repeat itself | dos_chat | why did my answer repeat itself | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | what should i do if the output looks wrong | base | what should i do if the output looks wrong | PASS |  | Retry with a shorter prompt or switch packs. |
| CHAT | what should i do if the output looks wrong | question_mark | what should i do if the output looks wrong | PASS |  | Retry with a shorter prompt or switch packs. |
| CHAT | what should i do if the output looks wrong | please | what should i do if the output looks wrong | PASS |  | Retry with a shorter prompt or switch packs. |
| CHAT | what should i do if the output looks wrong | short | what should i do if the output looks wrong | PASS |  | Retry with a shorter prompt or switch packs. |
| CHAT | what should i do if the output looks wrong | typed | what should i do if the output looks wrong | PASS |  | Retry with a shorter prompt or switch packs. |
| CHAT | what should i do if the output looks wrong | dos_chat | what should i do if the output looks wrong | PASS |  | Retry with a shorter prompt or switch packs. |
| CHAT | the answer sounds weird | base | the answer sounds weird | PASS |  | Retry with a shorter prompt or switch packs. Ask a follow-up and I can add more detail. |
| CHAT | the answer sounds weird | question_mark | the answer sounds weird | PASS |  | Retry with a shorter prompt or switch packs. Ask a follow-up and I can add more detail. |
| CHAT | the answer sounds weird | please | the answer sounds weird | PASS |  | Retry with a shorter prompt or switch packs. Ask a follow-up and I can add more detail. |
| CHAT | the answer sounds weird | short | the answer sounds weird | PASS |  | Retry with a shorter prompt or switch packs. Ask a follow-up and I can add more detail. |
| CHAT | the answer sounds weird | typed | the answer sounds weird | PASS |  | Retry with a shorter prompt or switch packs. Ask a follow-up and I can add more detail. |
| CHAT | the answer sounds weird | dos_chat | the answer sounds weird | PASS |  | Retry with a shorter prompt or switch packs. Ask a follow-up and I can add more detail. |
| CHAT | give me a three step bug fixing plan | base | give me a three step bug fixing plan | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | give me a three step bug fixing plan | question_mark | give me a three step bug fixing plan | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | give me a three step bug fixing plan | please | give me a three step bug fixing plan | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | give me a three step bug fixing plan | short | give me a three step bug fixing plan | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | give me a three step bug fixing plan | typed | give me a three step bug fixing plan | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | give me a three step bug fixing plan | dos_chat | give me a three step bug fixing plan | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | make a tiny plan for fixing a bug | base | make a tiny plan for fixing a bug | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | make a tiny plan for fixing a bug | question_mark | make a tiny plan for fixing a bug | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | make a tiny plan for fixing a bug | please | make a tiny plan for fixing a bug | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | make a tiny plan for fixing a bug | short | make a tiny plan for fixing a bug | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | make a tiny plan for fixing a bug | typed | make a tiny plan for fixing a bug | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | make a tiny plan for fixing a bug | dos_chat | make a tiny plan for fixing a bug | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | i feel stuck debugging this | base | i feel stuck debugging this | PASS |  | Check the first error, change one thing, then test again. Type a question and I will answer. |
| CHAT | i feel stuck debugging this | question_mark | i feel stuck debugging this | PASS |  | Check the first error, change one thing, then test again. Type a question and I will answer. |
| CHAT | i feel stuck debugging this | please | i feel stuck debugging this | PASS |  | Check the first error, change one thing, then test again. Type a question and I will answer. |
| CHAT | i feel stuck debugging this | short | i feel stuck debugging this | PASS |  | Check the first error, change one thing, then test again. Type a question and I will answer. |
| CHAT | i feel stuck debugging this | typed | i feel stuck debugging this | PASS |  | Check the first error, change one thing, then test again. Type a question and I will answer. |
| CHAT | i feel stuck debugging this | dos_chat | i feel stuck debugging this | PASS |  | Check the first error, change one thing, then test again. Type a question and I will answer. |
| CHAT | what is the difference between my prompt and your answer | base | what is the difference between my prompt and your answer | PASS |  | A prompt is what you type. An answer is the model output. That is the simple version. |
| CHAT | what is the difference between my prompt and your answer | question_mark | what is the difference between my prompt and your answer | PASS |  | A prompt is what you type. An answer is the model output. That is the simple version. |
| CHAT | what is the difference between my prompt and your answer | please | what is the difference between my prompt and your answer | PASS |  | A prompt is what you type. An answer is the model output. That is the simple version. |
| CHAT | what is the difference between my prompt and your answer | short | what is the difference between my prompt and your answer | PASS |  | A prompt is what you type. An answer is the model output. That is the simple version. |
| CHAT | what is the difference between my prompt and your answer | typed | what is the difference between my prompt and your answer | PASS |  | A prompt is what you type. An answer is the model output. That is the simple version. |
| CHAT | what is the difference between my prompt and your answer | dos_chat | what is the difference between my prompt and your answer | PASS |  | A prompt is what you type. An answer is the model output. That is the simple version. |
| CHAT | i am stuck and need a small next step | base | i am stuck and need a small next step | PASS |  | Choose one small step, test it, and then adjust. Check the first error, change one thing, then test again. |
| CHAT | i am stuck and need a small next step | question_mark | i am stuck and need a small next step | PASS |  | Choose one small step, test it, and then adjust. Check the first error, change one thing, then test again. |
| CHAT | i am stuck and need a small next step | please | i am stuck and need a small next step | PASS |  | Choose one small step, test it, and then adjust. Check the first error, change one thing, then test again. |
| CHAT | i am stuck and need a small next step | short | i am stuck and need a small next step | PASS |  | Choose one small step, test it, and then adjust. Check the first error, change one thing, then test again. |
| CHAT | i am stuck and need a small next step | typed | i am stuck and need a small next step | PASS |  | Choose one small step, test it, and then adjust. Check the first error, change one thing, then test again. |
| CHAT | i am stuck and need a small next step | dos_chat | i am stuck and need a small next step | PASS |  | Choose one small step, test it, and then adjust. Check the first error, change one thing, then test again. |
| CHAT | how do i focus for a few minutes | base | how do i focus for a few minutes | PASS |  | Remove one distraction and choose one small task. |
| CHAT | how do i focus for a few minutes | question_mark | how do i focus for a few minutes | PASS |  | Remove one distraction and choose one small task. |
| CHAT | how do i focus for a few minutes | please | how do i focus for a few minutes | PASS |  | Remove one distraction and choose one small task. |
| CHAT | how do i focus for a few minutes | short | how do i focus for a few minutes | PASS |  | Remove one distraction and choose one small task. |
| CHAT | how do i focus for a few minutes | typed | how do i focus for a few minutes | PASS |  | Remove one distraction and choose one small task. |
| CHAT | how do i focus for a few minutes | dos_chat | how do i focus for a few minutes | PASS |  | Remove one distraction and choose one small task. |
| CHAT | can you make a tiny plan for today | base | can you make a tiny plan for today | PASS |  | Pick one goal, list three steps, then start small. Ask one clear prompt and I will keep going. |
| CHAT | can you make a tiny plan for today | question_mark | can you make a tiny plan for today | PASS |  | Pick one goal, list three steps, then start small. Ask one clear prompt and I will keep going. |
| CHAT | can you make a tiny plan for today | please | can you make a tiny plan for today | PASS |  | Pick one goal, list three steps, then start small. Ask one clear prompt and I will keep going. |
| CHAT | can you make a tiny plan for today | short | can you make a tiny plan for today | PASS |  | Pick one goal, list three steps, then start small. Ask one clear prompt and I will keep going. |
| CHAT | can you make a tiny plan for today | typed | can you make a tiny plan for today | PASS |  | Pick one goal, list three steps, then start small. Ask one clear prompt and I will keep going. |
| CHAT | can you make a tiny plan for today | dos_chat | can you make a tiny plan for today | PASS |  | Pick one goal, list three steps, then start small. Ask one clear prompt and I will keep going. |
| CHAT | what should i do today | base | what should i do today | PASS |  | Pick one useful small task. Start small, then adjust after the first step. |
| CHAT | what should i do today | question_mark | what should i do today | PASS |  | Pick one useful small task. Start small, then adjust after the first step. |
| CHAT | what should i do today | please | what should i do today | PASS |  | Pick one useful small task. Start small, then adjust after the first step. |
| CHAT | what should i do today | short | what should i do today | PASS |  | Pick one useful small task. Start small, then adjust after the first step. |
| CHAT | what should i do today | typed | what should i do today | PASS |  | Pick one useful small task. Start small, then adjust after the first step. |
| CHAT | what should i do today | dos_chat | what should i do today | PASS |  | Pick one useful small task. Start small, then adjust after the first step. |
| CHAT | help me decide between two choices | base | help me decide between two choices | PASS |  | Name the choices and the goal, then choose one step. Start small, then adjust after the first step. |
| CHAT | help me decide between two choices | question_mark | help me decide between two choices | PASS |  | Name the choices and the goal, then choose one step. Start small, then adjust after the first step. |
| CHAT | help me decide between two choices | please | help me decide between two choices | PASS |  | Name the choices and the goal, then choose one step. Start small, then adjust after the first step. |
| CHAT | help me decide between two choices | short | help me decide between two choices | PASS |  | Name the choices and the goal, then choose one step. Start small, then adjust after the first step. |
| CHAT | help me decide between two choices | typed | help me decide between two choices | PASS |  | Name the choices and the goal, then choose one step. Start small, then adjust after the first step. |
| CHAT | help me decide between two choices | dos_chat | help me decide between two choices | PASS |  | Name the choices and the goal, then choose one step. Start small, then adjust after the first step. |
| CHAT | tell me whether the release is ready | base | tell me whether the release is ready | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | tell me whether the release is ready | question_mark | tell me whether the release is ready | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | tell me whether the release is ready | please | tell me whether the release is ready | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | tell me whether the release is ready | short | tell me whether the release is ready | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | tell me whether the release is ready | typed | tell me whether the release is ready | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | tell me whether the release is ready | dos_chat | tell me whether the release is ready | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | give me a release status update | base | give me a release status update | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | give me a release status update | question_mark | give me a release status update | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | give me a release status update | please | give me a release status update | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | give me a release status update | short | give me a release status update | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | give me a release status update | typed | give me a release status update | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | give me a release status update | dos_chat | give me a release status update | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | what should i verify before release | base | what should i verify before release | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | what should i verify before release | question_mark | what should i verify before release | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | what should i verify before release | please | what should i verify before release | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | what should i verify before release | short | what should i verify before release | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | what should i verify before release | typed | what should i verify before release | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | what should i verify before release | dos_chat | what should i verify before release | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | write one friendly sentence about this demo | base | write one friendly sentence about this demo | PASS |  | The demo is live and runs locally in DOS. Type a question and I will answer. |
| CHAT | write one friendly sentence about this demo | question_mark | write one friendly sentence about this demo | PASS |  | The demo is live and runs locally in DOS. Type a question and I will answer. |
| CHAT | write one friendly sentence about this demo | please | write one friendly sentence about this demo | PASS |  | The demo is live and runs locally in DOS. Type a question and I will answer. |
| CHAT | write one friendly sentence about this demo | short | write one friendly sentence about this demo | PASS |  | The demo is live and runs locally in DOS. Type a question and I will answer. |
| CHAT | write one friendly sentence about this demo | typed | write one friendly sentence about this demo | PASS |  | The demo is live and runs locally in DOS. Type a question and I will answer. |
| CHAT | write one friendly sentence about this demo | dos_chat | write one friendly sentence about this demo | PASS |  | The demo is live and runs locally in DOS. Type a question and I will answer. |
| CHAT | tell me a short story about a dos model | base | tell me a short story about a dos model | PASS |  | A tiny model woke up inside DOS and answered one prompt. It runs locally in this DOS demo. |
| CHAT | tell me a short story about a dos model | question_mark | tell me a short story about a dos model | PASS |  | A tiny model woke up inside DOS and answered one prompt. It runs locally in this DOS demo. |
| CHAT | tell me a short story about a dos model | please | tell me a short story about a dos model | PASS |  | A tiny model woke up inside DOS and answered one prompt. It runs locally in this DOS demo. |
| CHAT | tell me a short story about a dos model | short | tell me a short story about a dos model | PASS |  | A tiny model woke up inside DOS and answered one prompt. It runs locally in this DOS demo. |
| CHAT | tell me a short story about a dos model | typed | tell me a short story about a dos model | PASS |  | A tiny model woke up inside DOS and answered one prompt. It runs locally in this DOS demo. |
| CHAT | tell me a short story about a dos model | dos_chat | tell me a short story about a dos model | PASS |  | A tiny model woke up inside DOS and answered one prompt. It runs locally in this DOS demo. |
| CHAT | tell me a joke | base | tell me a joke | PASS |  | DOS smiled because it found its prompt. It runs locally in this DOS demo. |
| CHAT | tell me a joke | question_mark | tell me a joke | PASS |  | DOS smiled because it found its prompt. It runs locally in this DOS demo. |
| CHAT | tell me a joke | please | tell me a joke | PASS |  | DOS smiled because it found its prompt. It runs locally in this DOS demo. |
| CHAT | tell me a joke | short | tell me a joke | PASS |  | DOS smiled because it found its prompt. It runs locally in this DOS demo. |
| CHAT | tell me a joke | typed | tell me a joke | PASS |  | DOS smiled because it found its prompt. It runs locally in this DOS demo. |
| CHAT | tell me a joke | dos_chat | tell me a joke | PASS |  | DOS smiled because it found its prompt. It runs locally in this DOS demo. |
| CHAT | what is your favorite color | base | what is your favorite color | PASS |  | I like green phosphor. That is the simple version. |
| CHAT | what is your favorite color | question_mark | what is your favorite color | PASS |  | I like green phosphor. That is the simple version. |
| CHAT | what is your favorite color | please | what is your favorite color | PASS |  | I like green phosphor. That is the simple version. |
| CHAT | what is your favorite color | short | what is your favorite color | PASS |  | I like green phosphor. That is the simple version. |
| CHAT | what is your favorite color | typed | what is your favorite color | PASS |  | I like green phosphor. That is the simple version. |
| CHAT | what is your favorite color | dos_chat | what is your favorite color | PASS |  | I like green phosphor. That is the simple version. |
| CHAT | can we talk about games | base | can we talk about games | PASS |  | Yes, games are a fine topic. Ask one clear prompt and I will keep going. |
| CHAT | can we talk about games | question_mark | can we talk about games | PASS |  | Yes, games are a fine topic. Ask one clear prompt and I will keep going. |
| CHAT | can we talk about games | please | can we talk about games | PASS |  | Yes, games are a fine topic. Ask one clear prompt and I will keep going. |
| CHAT | can we talk about games | short | can we talk about games | PASS |  | Yes, games are a fine topic. Ask one clear prompt and I will keep going. |
| CHAT | can we talk about games | typed | can we talk about games | PASS |  | Yes, games are a fine topic. Ask one clear prompt and I will keep going. |
| CHAT | can we talk about games | dos_chat | can we talk about games | PASS |  | Yes, games are a fine topic. Ask one clear prompt and I will keep going. |
| CHAT | i feel worried | base | i feel worried | PASS |  | Name the worry, then choose one step. Notice the feeling, then choose one small step. |
| CHAT | i feel worried | question_mark | i feel worried | PASS |  | Name the worry, then choose one step. Notice the feeling, then choose one small step. |
| CHAT | i feel worried | please | i feel worried | PASS |  | Name the worry, then choose one step. Notice the feeling, then choose one small step. |
| CHAT | i feel worried | short | i feel worried | PASS |  | Name the worry, then choose one step. Notice the feeling, then choose one small step. |
| CHAT | i feel worried | typed | i feel worried | PASS |  | Name the worry, then choose one step. Notice the feeling, then choose one small step. |
| CHAT | i feel worried | dos_chat | i feel worried | PASS |  | Name the worry, then choose one step. Notice the feeling, then choose one small step. |
| CHAT | i am tired | base | i am tired | PASS |  | Rest if you can. Notice the feeling, then choose one small step. |
| CHAT | i am tired | question_mark | i am tired | PASS |  | Rest if you can. Notice the feeling, then choose one small step. |
| CHAT | i am tired | please | i am tired | PASS |  | Rest if you can. Notice the feeling, then choose one small step. |
| CHAT | i am tired | short | i am tired | PASS |  | Rest if you can. Notice the feeling, then choose one small step. |
| CHAT | i am tired | typed | i am tired | PASS |  | Rest if you can. Notice the feeling, then choose one small step. |
| CHAT | i am tired | dos_chat | i am tired | PASS |  | Rest if you can. Notice the feeling, then choose one small step. |
| CHAT | i feel lonely | base | i feel lonely | PASS |  | I can keep you company briefly. Notice the feeling, then choose one small step. |
| CHAT | i feel lonely | question_mark | i feel lonely | PASS |  | I can keep you company briefly. Notice the feeling, then choose one small step. |
| CHAT | i feel lonely | please | i feel lonely | PASS |  | I can keep you company briefly. Notice the feeling, then choose one small step. |
| CHAT | i feel lonely | short | i feel lonely | PASS |  | I can keep you company briefly. Notice the feeling, then choose one small step. |
| CHAT | i feel lonely | typed | i feel lonely | PASS |  | I can keep you company briefly. Notice the feeling, then choose one small step. |
| CHAT | i feel lonely | dos_chat | i feel lonely | PASS |  | I can keep you company briefly. Notice the feeling, then choose one small step. |
| CHAT | could you walk me through a simple troubleshooting process | base | could you walk me through a simple troubleshooting process | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | could you walk me through a simple troubleshooting process | question_mark | could you walk me through a simple troubleshooting process | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | could you walk me through a simple troubleshooting process | please | could you walk me through a simple troubleshooting process | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | could you walk me through a simple troubleshooting process | short | could you walk me through a simple troubleshooting process | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | could you walk me through a simple troubleshooting process | typed | could you walk me through a simple troubleshooting process | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | could you walk me through a simple troubleshooting process | dos_chat | could you walk me through a simple troubleshooting process | PASS |  | Check the first error, change one thing, then test again. |
| CHAT | help me pick a next task when i am overwhelmed | base | help me pick a next task when i am overwhelmed | PASS |  | Choose one small task, then take the first step. |
| CHAT | help me pick a next task when i am overwhelmed | question_mark | help me pick a next task when i am overwhelmed | PASS |  | Choose one small task, then take the first step. |
| CHAT | help me pick a next task when i am overwhelmed | please | help me pick a next task when i am overwhelmed | PASS |  | Choose one small task, then take the first step. |
| CHAT | help me pick a next task when i am overwhelmed | short | help me pick a next task when i am overwhelmed | PASS |  | Choose one small task, then take the first step. |
| CHAT | help me pick a next task when i am overwhelmed | typed | help me pick a next task when i am overwhelmed | PASS |  | Choose one small task, then take the first step. |
| CHAT | help me pick a next task when i am overwhelmed | dos_chat | help me pick a next task when i am overwhelmed | PASS |  | Choose one small task, then take the first step. |
| CHAT | why should i trust this demo | base | why should i trust this demo | PASS |  | The demo uses local model weights in DOS. Type a question and I will answer. |
| CHAT | why should i trust this demo | question_mark | why should i trust this demo | PASS |  | The demo uses local model weights in DOS. Type a question and I will answer. |
| CHAT | why should i trust this demo | please | why should i trust this demo | PASS |  | The demo uses local model weights in DOS. Type a question and I will answer. |
| CHAT | why should i trust this demo | short | why should i trust this demo | PASS |  | The demo uses local model weights in DOS. Type a question and I will answer. |
| CHAT | why should i trust this demo | typed | why should i trust this demo | PASS |  | The demo uses local model weights in DOS. Type a question and I will answer. |
| CHAT | why should i trust this demo | dos_chat | why should i trust this demo | PASS |  | The demo uses local model weights in DOS. Type a question and I will answer. |
| CHAT | what happens if i ask a long question | base | what happens if i ask a long question | PASS |  | Short prompts work better in this DOS demo. |
| CHAT | what happens if i ask a long question | question_mark | what happens if i ask a long question | PASS |  | Short prompts work better in this DOS demo. |
| CHAT | what happens if i ask a long question | please | what happens if i ask a long question | PASS |  | Short prompts work better in this DOS demo. |
| CHAT | what happens if i ask a long question | short | what happens if i ask a long question | PASS |  | Short prompts work better in this DOS demo. |
| CHAT | what happens if i ask a long question | typed | what happens if i ask a long question | PASS |  | Short prompts work better in this DOS demo. |
| CHAT | what happens if i ask a long question | dos_chat | what happens if i ask a long question | PASS |  | Short prompts work better in this DOS demo. |
| CHAT | can you answer questions about history | base | can you answer questions about history | PASS |  | Ask one simple question about the topic. Type a question and I will answer. |
| CHAT | can you answer questions about history | question_mark | can you answer questions about history | PASS |  | Ask one simple question about the topic. Type a question and I will answer. |
| CHAT | can you answer questions about history | please | can you answer questions about history | PASS |  | Ask one simple question about the topic. Type a question and I will answer. |
| CHAT | can you answer questions about history | short | can you answer questions about history | PASS |  | Ask one simple question about the topic. Type a question and I will answer. |
| CHAT | can you answer questions about history | typed | can you answer questions about history | PASS |  | Ask one simple question about the topic. Type a question and I will answer. |
| CHAT | can you answer questions about history | dos_chat | can you answer questions about history | PASS |  | Ask one simple question about the topic. Type a question and I will answer. |
| CHAT | write a friendly note about an old computer | base | write a friendly note about an old computer | PASS |  | This old DOS computer can run a friendly local model. |
| CHAT | write a friendly note about an old computer | question_mark | write a friendly note about an old computer | PASS |  | This old DOS computer can run a friendly local model. |
| CHAT | write a friendly note about an old computer | please | write a friendly note about an old computer | PASS |  | This old DOS computer can run a friendly local model. |
| CHAT | write a friendly note about an old computer | short | write a friendly note about an old computer | PASS |  | This old DOS computer can run a friendly local model. |
| CHAT | write a friendly note about an old computer | typed | write a friendly note about an old computer | PASS |  | This old DOS computer can run a friendly local model. |
| CHAT | write a friendly note about an old computer | dos_chat | write a friendly note about an old computer | PASS |  | This old DOS computer can run a friendly local model. |
| CHAT | what does local mean here | base | what does local mean here | PASS |  | Local means the model runs on this machine. Type a question and I will answer. |
| CHAT | what does local mean here | question_mark | what does local mean here | PASS |  | Local means the model runs on this machine. Type a question and I will answer. |
| CHAT | what does local mean here | please | what does local mean here | PASS |  | Local means the model runs on this machine. Type a question and I will answer. |
| CHAT | what does local mean here | short | what does local mean here | PASS |  | Local means the model runs on this machine. Type a question and I will answer. |
| CHAT | what does local mean here | typed | what does local mean here | PASS |  | Local means the model runs on this machine. Type a question and I will answer. |
| CHAT | what does local mean here | dos_chat | what does local mean here | PASS |  | Local means the model runs on this machine. Type a question and I will answer. |
| CHAT | is there any memory between turns | base | is there any memory between turns | PASS |  | I remember only small facts during this session. |
| CHAT | is there any memory between turns | question_mark | is there any memory between turns | PASS |  | I remember only small facts during this session. |
| CHAT | is there any memory between turns | please | is there any memory between turns | PASS |  | I remember only small facts during this session. |
| CHAT | is there any memory between turns | short | is there any memory between turns | PASS |  | I remember only small facts during this session. |
| CHAT | is there any memory between turns | typed | is there any memory between turns | PASS |  | I remember only small facts during this session. |
| CHAT | is there any memory between turns | dos_chat | is there any memory between turns | PASS |  | I remember only small facts during this session. |
| CHAT | how do i avoid repeated answers | base | how do i avoid repeated answers | PASS |  | Use a shorter prompt and reset if the answer loops. |
| CHAT | how do i avoid repeated answers | question_mark | how do i avoid repeated answers | PASS |  | Use a shorter prompt and reset if the answer loops. |
| CHAT | how do i avoid repeated answers | please | how do i avoid repeated answers | PASS |  | Use a shorter prompt and reset if the answer loops. |
| CHAT | how do i avoid repeated answers | short | how do i avoid repeated answers | PASS |  | Use a shorter prompt and reset if the answer loops. |
| CHAT | how do i avoid repeated answers | typed | how do i avoid repeated answers | PASS |  | Use a shorter prompt and reset if the answer loops. |
| CHAT | how do i avoid repeated answers | dos_chat | how do i avoid repeated answers | PASS |  | Use a shorter prompt and reset if the answer loops. |
| CHAT | give me a tiny checklist for release | base | give me a tiny checklist for release | PASS |  | Check the tag, assets, checksums, and tests. |
| CHAT | give me a tiny checklist for release | question_mark | give me a tiny checklist for release | PASS |  | Check the tag, assets, checksums, and tests. |
| CHAT | give me a tiny checklist for release | please | give me a tiny checklist for release | PASS |  | Check the tag, assets, checksums, and tests. |
| CHAT | give me a tiny checklist for release | short | give me a tiny checklist for release | PASS |  | Check the tag, assets, checksums, and tests. |
| CHAT | give me a tiny checklist for release | typed | give me a tiny checklist for release | PASS |  | Check the tag, assets, checksums, and tests. |
| CHAT | give me a tiny checklist for release | dos_chat | give me a tiny checklist for release | PASS |  | Check the tag, assets, checksums, and tests. |
| CHAT | what is an emulator | base | what is an emulator | PASS |  | An emulator runs one machine inside another. |
| CHAT | what is an emulator | question_mark | what is an emulator | PASS |  | An emulator runs one machine inside another. |
| CHAT | what is an emulator | please | what is an emulator | PASS |  | An emulator runs one machine inside another. |
| CHAT | what is an emulator | short | what is an emulator | PASS |  | An emulator runs one machine inside another. |
| CHAT | what is an emulator | typed | what is an emulator | PASS |  | An emulator runs one machine inside another. |
| CHAT | what is an emulator | dos_chat | what is an emulator | PASS |  | An emulator runs one machine inside another. |
| CHAT | why no internet | base | why no internet | PASS |  | DOS cannot browse here; the answer comes from local files. |
| CHAT | why no internet | question_mark | why no internet | PASS |  | DOS cannot browse here; the answer comes from local files. |
| CHAT | why no internet | please | why no internet | PASS |  | DOS cannot browse here; the answer comes from local files. |
| CHAT | why no internet | short | why no internet | PASS |  | DOS cannot browse here; the answer comes from local files. |
| CHAT | why no internet | typed | why no internet | PASS |  | DOS cannot browse here; the answer comes from local files. |
| CHAT | why no internet | dos_chat | why no internet | PASS |  | DOS cannot browse here; the answer comes from local files. |
| DOSHELP | why does my protected mode program need cwsdpmi | base | why does my protected mode program need cwsdpmi | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does my protected mode program need cwsdpmi | question_mark | why does my protected mode program need cwsdpmi | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does my protected mode program need cwsdpmi | please | why does my protected mode program need cwsdpmi | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does my protected mode program need cwsdpmi | short | why does my protected mode program need cwsdpmi | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does my protected mode program need cwsdpmi | typed | why does my protected mode program need cwsdpmi | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does my protected mode program need cwsdpmi | dos_chat | why does my protected mode program need cwsdpmi | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does dos need a dpmi host | base | why does dos need a dpmi host | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does dos need a dpmi host | question_mark | why does dos need a dpmi host | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does dos need a dpmi host | please | why does dos need a dpmi host | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does dos need a dpmi host | short | why does dos need a dpmi host | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does dos need a dpmi host | typed | why does dos need a dpmi host | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | why does dos need a dpmi host | dos_chat | why does dos need a dpmi host | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | how do i leave more conventional memory free | base | how do i leave more conventional memory free | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i leave more conventional memory free | question_mark | how do i leave more conventional memory free | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i leave more conventional memory free | please | how do i leave more conventional memory free | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i leave more conventional memory free | short | how do i leave more conventional memory free | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i leave more conventional memory free | typed | how do i leave more conventional memory free | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i leave more conventional memory free | dos_chat | how do i leave more conventional memory free | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | what should i put in config.sys for this assistant | base | what should i put in config.sys for this assistant | PASS |  | Use HIMEM, DOS HIGH UMB, FILES 30, and BUFFERS 20 as the safe CONFIG SYS baseline. |
| DOSHELP | what should i put in config.sys for this assistant | question_mark | what should i put in config.sys for this assistant | PASS |  | Use HIMEM, DOS HIGH UMB, FILES 30, and BUFFERS 20 as the safe CONFIG SYS baseline. |
| DOSHELP | what should i put in config.sys for this assistant | please | what should i put in config.sys for this assistant | PASS |  | Use HIMEM, DOS HIGH UMB, FILES 30, and BUFFERS 20 as the safe CONFIG SYS baseline. |
| DOSHELP | what should i put in config.sys for this assistant | short | what should i put in config.sys for this assistant | PASS |  | Use HIMEM, DOS HIGH UMB, FILES 30, and BUFFERS 20 as the safe CONFIG SYS baseline. |
| DOSHELP | what should i put in config.sys for this assistant | typed | what should i put in config.sys for this assistant | PASS |  | Use HIMEM, DOS HIGH UMB, FILES 30, and BUFFERS 20 as the safe CONFIG SYS baseline. |
| DOSHELP | what should i put in config.sys for this assistant | dos_chat | what should i put in config.sys for this assistant | PASS |  | Use HIMEM, DOS HIGH UMB, FILES 30, and BUFFERS 20 as the safe CONFIG SYS baseline. |
| DOSHELP | how do i tune config.sys memory | base | how do i tune config.sys memory | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i tune config.sys memory | question_mark | how do i tune config.sys memory | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i tune config.sys memory | please | how do i tune config.sys memory | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i tune config.sys memory | short | how do i tune config.sys memory | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i tune config.sys memory | typed | how do i tune config.sys memory | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | how do i tune config.sys memory | dos_chat | how do i tune config.sys memory | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | my autoexec is messy and slow | base | my autoexec is messy and slow | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | my autoexec is messy and slow | question_mark | my autoexec is messy and slow | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | my autoexec is messy and slow | please | my autoexec is messy and slow | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | my autoexec is messy and slow | short | my autoexec is messy and slow | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | my autoexec is messy and slow | typed | my autoexec is messy and slow | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | my autoexec is messy and slow | dos_chat | my autoexec is messy and slow | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | how should i clean autoexec.bat | base | how should i clean autoexec.bat | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | how should i clean autoexec.bat | question_mark | how should i clean autoexec.bat | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | how should i clean autoexec.bat | please | how should i clean autoexec.bat | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | how should i clean autoexec.bat | short | how should i clean autoexec.bat | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | how should i clean autoexec.bat | typed | how should i clean autoexec.bat | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | how should i clean autoexec.bat | dos_chat | how should i clean autoexec.bat | PASS |  | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | write a safe batch check for the model directory | base | write a safe batch check for the model directory | PASS |  | In a batch file, use IF EXIST checks, clear status messages, and 8-dot-3 names for model files. |
| DOSHELP | write a safe batch check for the model directory | question_mark | write a safe batch check for the model directory | PASS |  | In a batch file, use IF EXIST checks, clear status messages, and 8-dot-3 names for model files. |
| DOSHELP | write a safe batch check for the model directory | please | write a safe batch check for the model directory | PASS |  | In a batch file, use IF EXIST checks, clear status messages, and 8-dot-3 names for model files. |
| DOSHELP | write a safe batch check for the model directory | short | write a safe batch check for the model directory | PASS |  | In a batch file, use IF EXIST checks, clear status messages, and 8-dot-3 names for model files. |
| DOSHELP | write a safe batch check for the model directory | typed | write a safe batch check for the model directory | PASS |  | In a batch file, use IF EXIST checks, clear status messages, and 8-dot-3 names for model files. |
| DOSHELP | write a safe batch check for the model directory | dos_chat | write a safe batch check for the model directory | PASS |  | In a batch file, use IF EXIST checks, clear status messages, and 8-dot-3 names for model files. |
| DOSHELP | what does if exist do in a batch file | base | what does if exist do in a batch file | PASS |  | In a batch file, IF EXIST checks whether a file or directory is present before running a command. |
| DOSHELP | what does if exist do in a batch file | question_mark | what does if exist do in a batch file | PASS |  | In a batch file, IF EXIST checks whether a file or directory is present before running a command. |
| DOSHELP | what does if exist do in a batch file | please | what does if exist do in a batch file | PASS |  | In a batch file, IF EXIST checks whether a file or directory is present before running a command. |
| DOSHELP | what does if exist do in a batch file | short | what does if exist do in a batch file | PASS |  | In a batch file, IF EXIST checks whether a file or directory is present before running a command. |
| DOSHELP | what does if exist do in a batch file | typed | what does if exist do in a batch file | PASS |  | In a batch file, IF EXIST checks whether a file or directory is present before running a command. |
| DOSHELP | what does if exist do in a batch file | dos_chat | what does if exist do in a batch file | PASS |  | In a batch file, IF EXIST checks whether a file or directory is present before running a command. |
| DOSHELP | where should model packs point | base | where should model packs point | PASS |  | Point MODEL at C:\MODEL or a pack-local checkpoint so the assistant can switch model paths. |
| DOSHELP | where should model packs point | question_mark | where should model packs point | PASS |  | Point MODEL at C:\MODEL or a pack-local checkpoint so the assistant can switch model paths. |
| DOSHELP | where should model packs point | please | where should model packs point | PASS |  | Point MODEL at C:\MODEL or a pack-local checkpoint so the assistant can switch model paths. |
| DOSHELP | where should model packs point | short | where should model packs point | PASS |  | Point MODEL at C:\MODEL or a pack-local checkpoint so the assistant can switch model paths. |
| DOSHELP | where should model packs point | typed | where should model packs point | PASS |  | Point MODEL at C:\MODEL or a pack-local checkpoint so the assistant can switch model paths. |
| DOSHELP | where should model packs point | dos_chat | where should model packs point | PASS |  | Point MODEL at C:\MODEL or a pack-local checkpoint so the assistant can switch model paths. |
| OFFICE | rewrite this politely: the artifact failed | base | rewrite this politely: the artifact failed | PASS |  | Use direct, polite wording, keep the artifact fact, and end with the next action. |
| OFFICE | rewrite this politely: the artifact failed | question_mark | rewrite this politely: the artifact failed | PASS |  | Use direct, polite wording, keep the artifact fact, and end with the next action. |
| OFFICE | rewrite this politely: the artifact failed | please | rewrite this politely: the artifact failed | PASS |  | Use direct, polite wording, keep the artifact fact, and end with the next action. |
| OFFICE | rewrite this politely: the artifact failed | short | rewrite this politely: the artifact failed | PASS |  | Use direct, polite wording, keep the artifact fact, and end with the next action. |
| OFFICE | rewrite this politely: the artifact failed | typed | rewrite this politely: the artifact failed | PASS |  | Use direct, polite wording, keep the artifact fact, and end with the next action. |
| OFFICE | rewrite this politely: the artifact failed | dos_chat | rewrite this politely: the artifact failed | PASS |  | Use direct, polite wording, keep the artifact fact, and end with the next action. |
| OFFICE | make this sentence sound professional: the release broke | base | make this sentence sound professional: the release broke | PASS |  | Use direct, polite, professional wording, name the release problem, and end with the next action. |
| OFFICE | make this sentence sound professional: the release broke | question_mark | make this sentence sound professional: the release broke | PASS |  | Use direct, polite, professional wording, name the release problem, and end with the next action. |
| OFFICE | make this sentence sound professional: the release broke | please | make this sentence sound professional: the release broke | PASS |  | Use direct, polite, professional wording, name the release problem, and end with the next action. |
| OFFICE | make this sentence sound professional: the release broke | short | make this sentence sound professional: the release broke | PASS |  | Use direct, polite, professional wording, name the release problem, and end with the next action. |
| OFFICE | make this sentence sound professional: the release broke | typed | make this sentence sound professional: the release broke | PASS |  | Use direct, polite, professional wording, name the release problem, and end with the next action. |
| OFFICE | make this sentence sound professional: the release broke | dos_chat | make this sentence sound professional: the release broke | PASS |  | Use direct, polite, professional wording, name the release problem, and end with the next action. |
| OFFICE | summarize this: tests passed but the tag was stale | base | summarize this: tests passed but the tag was stale | PASS |  | Summary: tests passed, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | summarize this: tests passed but the tag was stale | question_mark | summarize this: tests passed but the tag was stale | PASS |  | Summary: tests passed, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | summarize this: tests passed but the tag was stale | please | summarize this: tests passed but the tag was stale | PASS |  | Summary: tests passed, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | summarize this: tests passed but the tag was stale | short | summarize this: tests passed but the tag was stale | PASS |  | Summary: tests passed, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | summarize this: tests passed but the tag was stale | typed | summarize this: tests passed but the tag was stale | PASS |  | Summary: tests passed, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | summarize this: tests passed but the tag was stale | dos_chat | summarize this: tests passed but the tag was stale | PASS |  | Summary: tests passed, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | summarize: tests passed but dosbox needed a helper file | base | summarize: tests passed but dosbox needed a helper file | PASS |  | Summary: tests passed, DOSBox need a helper file. |
| OFFICE | summarize: tests passed but dosbox needed a helper file | question_mark | summarize: tests passed but dosbox needed a helper file | PASS |  | Summary: tests passed, DOSBox need a helper file. |
| OFFICE | summarize: tests passed but dosbox needed a helper file | please | summarize: tests passed but dosbox needed a helper file | PASS |  | Summary: tests passed, DOSBox need a helper file. |
| OFFICE | summarize: tests passed but dosbox needed a helper file | short | summarize: tests passed but dosbox needed a helper file | PASS |  | Summary: tests passed, DOSBox need a helper file. |
| OFFICE | summarize: tests passed but dosbox needed a helper file | typed | summarize: tests passed but dosbox needed a helper file | PASS |  | Summary: tests passed, DOSBox need a helper file. |
| OFFICE | summarize: tests passed but dosbox needed a helper file | dos_chat | summarize: tests passed but dosbox needed a helper file | PASS |  | Summary: tests passed, DOSBox need a helper file. |
| OFFICE | make this clearer: checksums changed after rebuild | base | make this clearer: checksums changed after rebuild | PASS |  | State that checksums changed after rebuild, why it matters, and the next action. |
| OFFICE | make this clearer: checksums changed after rebuild | question_mark | make this clearer: checksums changed after rebuild | PASS |  | State that checksums changed after rebuild, why it matters, and the next action. |
| OFFICE | make this clearer: checksums changed after rebuild | please | make this clearer: checksums changed after rebuild | PASS |  | State that checksums changed after rebuild, why it matters, and the next action. |
| OFFICE | make this clearer: checksums changed after rebuild | short | make this clearer: checksums changed after rebuild | PASS |  | State that checksums changed after rebuild, why it matters, and the next action. |
| OFFICE | make this clearer: checksums changed after rebuild | typed | make this clearer: checksums changed after rebuild | PASS |  | State that checksums changed after rebuild, why it matters, and the next action. |
| OFFICE | make this clearer: checksums changed after rebuild | dos_chat | make this clearer: checksums changed after rebuild | PASS |  | State that checksums changed after rebuild, why it matters, and the next action. |
| OFFICE | make this clearer: the artifact uploaded but the tag was stale | base | make this clearer: the artifact uploaded but the tag was stale | PASS |  | State that the artifact uploaded, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | make this clearer: the artifact uploaded but the tag was stale | question_mark | make this clearer: the artifact uploaded but the tag was stale | PASS |  | State that the artifact uploaded, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | make this clearer: the artifact uploaded but the tag was stale | please | make this clearer: the artifact uploaded but the tag was stale | PASS |  | State that the artifact uploaded, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | make this clearer: the artifact uploaded but the tag was stale | short | make this clearer: the artifact uploaded but the tag was stale | PASS |  | State that the artifact uploaded, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | make this clearer: the artifact uploaded but the tag was stale | typed | make this clearer: the artifact uploaded but the tag was stale | PASS |  | State that the artifact uploaded, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | make this clearer: the artifact uploaded but the tag was stale | dos_chat | make this clearer: the artifact uploaded but the tag was stale | PASS |  | State that the artifact uploaded, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | shorten this sentence without losing the decision | base | shorten this sentence without losing the decision | PASS |  | Keep the decision and next action, then remove repeated explanation. |
| OFFICE | shorten this sentence without losing the decision | question_mark | shorten this sentence without losing the decision | PASS |  | Keep the decision and next action, then remove repeated explanation. |
| OFFICE | shorten this sentence without losing the decision | please | shorten this sentence without losing the decision | PASS |  | Keep the decision and next action, then remove repeated explanation. |
| OFFICE | shorten this sentence without losing the decision | short | shorten this sentence without losing the decision | PASS |  | Keep the decision and next action, then remove repeated explanation. |
| OFFICE | shorten this sentence without losing the decision | typed | shorten this sentence without losing the decision | PASS |  | Keep the decision and next action, then remove repeated explanation. |
| OFFICE | shorten this sentence without losing the decision | dos_chat | shorten this sentence without losing the decision | PASS |  | Keep the decision and next action, then remove repeated explanation. |
| OFFICE | shorten: we need to verify the release before publishing | base | shorten: we need to verify the release before publishing | PASS |  | Short version: verify the release before publishing. |
| OFFICE | shorten: we need to verify the release before publishing | question_mark | shorten: we need to verify the release before publishing | PASS |  | Short version: verify the release before publishing. |
| OFFICE | shorten: we need to verify the release before publishing | please | shorten: we need to verify the release before publishing | PASS |  | Short version: verify the release before publishing. |
| OFFICE | shorten: we need to verify the release before publishing | short | shorten: we need to verify the release before publishing | PASS |  | Short version: verify the release before publishing. |
| OFFICE | shorten: we need to verify the release before publishing | typed | shorten: we need to verify the release before publishing | PASS |  | Short version: verify the release before publishing. |
| OFFICE | shorten: we need to verify the release before publishing | dos_chat | shorten: we need to verify the release before publishing | PASS |  | Short version: verify the release before publishing. |
| OFFICE | write a status update about a delayed build | base | write a status update about a delayed build | PASS |  | Use direct, polite, concrete wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a status update about a delayed build | question_mark | write a status update about a delayed build | PASS |  | Use direct, polite, concrete wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a status update about a delayed build | please | write a status update about a delayed build | PASS |  | Use direct, polite, concrete wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a status update about a delayed build | short | write a status update about a delayed build | PASS |  | Use direct, polite, concrete wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a status update about a delayed build | typed | write a status update about a delayed build | PASS |  | Use direct, polite, concrete wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a status update about a delayed build | dos_chat | write a status update about a delayed build | PASS |  | Use direct, polite, concrete wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a polite status update about a delayed build | base | write a polite status update about a delayed build | PASS |  | Use direct polite wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a polite status update about a delayed build | question_mark | write a polite status update about a delayed build | PASS |  | Use direct polite wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a polite status update about a delayed build | please | write a polite status update about a delayed build | PASS |  | Use direct polite wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a polite status update about a delayed build | short | write a polite status update about a delayed build | PASS |  | Use direct polite wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a polite status update about a delayed build | typed | write a polite status update about a delayed build | PASS |  | Use direct polite wording, include the current blocker, and end with the next action or timing. |
| OFFICE | write a polite status update about a delayed build | dos_chat | write a polite status update about a delayed build | PASS |  | Use direct polite wording, include the current blocker, and end with the next action or timing. |
