# Assistant Pack Quality Report: CHAT

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab4096`
Evaluation backend: `float`
Quality suite: `assistant-pack`
Quality status: `NEEDS_TRAINING`
Average score: `0.904`
Prompt pass rate: `53/64` at threshold `0.90`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| chat_hello | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_how are you | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_what can you do | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_limits | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_hi | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_good_morning | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_how_are_you | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_are_you | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_can_you_do | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_we_talk | 0.798 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_does_prompt_mean | 1.000 | 4 | 0.0% | 1 | 0 | yes | PASS |
| chat_explain_prompt | 1.000 | 4 | 0.0% | 1 | 0 | yes | PASS |
| chat_idea | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_help_me | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_can_you_use_the_internet | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_is_this_scripted | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_is_the_answer_scripted | 1.000 | 1 | 0.0% | 1 | 0 | yes | PASS |
| chat_do_you_use_a_script | 1.000 | 1 | 0.0% | 1 | 0 | yes | PASS |
| chat_is_this_actual_inference | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_is_the_demo_live | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_basic | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_inference | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_why_are_answers_short | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_thank_you | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_goodbye | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_i_am_tired | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_i_am_worried | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_i_feel_good | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_what_should_i_do_next | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_help_me_decide | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_do_you_remember_me | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_tell_me_a_story | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_we_talk_about_music | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_your_favorite_color | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_do_you_do_for_fun | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_are_you_alive | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_do_you_have_feelings | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_can_you_write | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_you_make_a_plan | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_a_good_habit | 0.759 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_how_do_i_relax | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_music | 1.000 | 3 | 0.0% | 1 | 0 | yes | PASS |
| chat_what_is_a_story | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_a_goal | 1.000 | 4 | 0.0% | 1 | 0 | yes | PASS |
| chat_what_is_work | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_life | 1.000 | 6 | 0.0% | 1 | 0 | yes | PASS |
| chat_good_night | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_i_like_this_demo | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_this_is_weird | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_ask_me_a_question | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_do_you_understand_english | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_i_am_feeling_bored | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_suggest_something_to_do | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_why_is_this_answer_looping | 1.000 | 5 | 0.0% | 1 | 0 | yes | PASS |
| chat_does_this_use_real_model_weights | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_should_i_do_if_the_output_looks | 1.000 | 4 | 0.0% | 1 | 0 | yes | PASS |
| chat_give_me_a_three_step_bug_fixing_plan | 0.904 | 6 | 9.1% | 2 | 0 | yes | PASS |
| chat_why_does_an_old_computer_demo_matter | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_tell_me_whether_the_release_is_ready | 1.000 | 5 | 0.0% | 2 | 0 | yes | PASS |
| chat_how_do_i_focus | 1.000 | 5 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_you_make_a_tiny_plan_for_today | 1.000 | 6 | 0.0% | 2 | 0 | yes | PASS |
| chat_explain_token | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_how_do_i_focus_for_a_few_minutes | 1.000 | 5 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_you_use_the_network | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### chat_hello

Prompt: `User: Hello. Note: Starting a conversation: Hello from DOS. Assistant:`

```text
Hello from DOS. Type a question and I will answer.
```

### chat_how are you

Prompt: `User: How are you? Note: Status check: I answer in DOS. Assistant:`

```text
I answer in DOS. It runs locally in this DOS demo.
```

### chat_what can you do

Prompt: `User: Hello, what can you do? Note: Assistant abilities: I chat in DOS. Assistant:`

```text
I chat in DOS. It runs locally in this DOS demo.
```

### chat_limits

Prompt: `User: What are your limits? Note: Model limits: I am tiny and local. Assistant:`

```text
I am tiny and local. This tiny session has clear limits.
```

### chat_hi

Prompt: `User: please answer this: hi Assistant:`

```text
Ask me a short question. Ask a follow-up and I can add more detail.
```

### chat_good_morning

Prompt: `User: short answer please: good morning Assistant:`

```text
Hello from DOS. Type a question and I will answer.
```

### chat_how_are_you

Prompt: `User: i typed this question: how are you Assistant:`

```text
I answer in DOS. It runs locally in this DOS demo.
```

### chat_what_are_you

Prompt: `User: dos chat question: what are you Assistant:`

```text
I am GPT2-BASIC. Ask a follow-up and I can add more detail.
```

### chat_what_can_you_do

Prompt: `User: help me with this: what can you do Assistant:`

```text
I can chat in DOS. It runs locally in this DOS demo.
```

### chat_can_we_talk

Prompt: `User: please answer this: can we talk Assistant:`

```text
Yes, games are a fine topic. Ask one clear prompt and I will keep going.
```

### chat_what_does_prompt_mean

Prompt: `User: short answer please: what does prompt mean Assistant:`

```text
A prompt is your typed question. That is the simple version.
```

### chat_explain_prompt

Prompt: `User: i typed this question: explain prompt Assistant:`

```text
A prompt is your typed question. That is the simple version.
```

### chat_idea

Prompt: `User: dos chat question: idea Assistant:`

```text
Try one clear question. Ask a follow-up and I can add more detail.
```

### chat_help_me

Prompt: `User: help me with this: help me Assistant:`

```text
I am tiny and local. This tiny session has clear limits.
```

### chat_can_you_use_the_internet

Prompt: `User: please answer this: can you use the internet Assistant:`

```text
I cannot use the internet. Ask one clear prompt and I will keep going.
```

### chat_is_this_scripted

Prompt: `User: short answer please: is this scripted Assistant:`

```text
No, it is real. Type a question and I will answer.
```

### chat_is_the_answer_scripted

Prompt: `User: i typed this question: is the answer scripted Assistant:`

```text
No, it is real. The answer comes from local model weights.
```

### chat_do_you_use_a_script

Prompt: `User: dos chat question: do you use a script Assistant:`

```text
No, it is real. The answer comes from local model weights.
```

### chat_is_this_actual_inference

Prompt: `User: help me with this: is this actual inference Assistant:`

```text
Yes, ask one practical question. Ask one clear prompt and I will keep going.
```

### chat_is_the_demo_live

Prompt: `User: please answer this: is the demo live Assistant:`

```text
The demo is live. It runs locally in this DOS demo.
```

### chat_what_is_basic

Prompt: `User: short answer please: what is basic Assistant:`

```text
BASIC is a simple programming language. It runs locally in this DOS demo.
```

### chat_what_is_inference

Prompt: `User: i typed this question: what is inference Assistant:`

```text
Yes, ask one practical question. Ask one clear prompt and I will keep going.
```

### chat_why_are_answers_short

Prompt: `User: dos chat question: why are answers short Assistant:`

```text
Short answers run faster in DOS. It runs locally in this DOS demo.
```

### chat_thank_you

Prompt: `User: help me with this: thank you Assistant:`

```text
You are welcome. Ask a follow-up and I can add more detail.
```

### chat_goodbye

Prompt: `User: please answer this: goodbye Assistant:`

```text
Yes, ask one practical question. Ask one clear prompt and I will keep going.
```

### chat_i_am_tired

Prompt: `User: short answer please: i am tired Assistant:`

```text
Hello from DOS. Type a question and I will answer.
```

### chat_i_am_worried

Prompt: `User: i typed this question: i am worried Assistant:`

```text
Yes, ask about one word. Ask one clear prompt and I will keep going.
```

### chat_i_feel_good

Prompt: `User: dos chat question: i feel good Assistant:`

```text
Hello from DOS. Type a question and I will answer.
```

### chat_what_should_i_do_next

Prompt: `User: help me with this: what should i do next Assistant:`

```text
Choose one clear next step. Start small, then adjust after the first step.
```

### chat_help_me_decide

Prompt: `User: please answer this: help me decide Assistant:`

```text
Name the choices and the goal. Start small, then adjust after the first step.
```

### chat_do_you_remember_me

Prompt: `User: short answer please: do you remember me Assistant:`

```text
Only within this session. This tiny session has clear limits.
```

### chat_tell_me_a_story

Prompt: `User: i typed this question: tell me a story Assistant:`

```text
A tiny model woke up inside DOS and answered one prompt. It runs locally in this DOS demo.
```

### chat_can_we_talk_about_music

Prompt: `User: dos chat question: can we talk about music Assistant:`

```text
Yes, music is a fine topic. Ask one clear prompt and I will keep going.
```

### chat_what_is_your_favorite_color

Prompt: `User: help me with this: what is your favorite color Assistant:`

```text
I like green phosphor. That is the simple version.
```

### chat_what_do_you_do_for_fun

Prompt: `User: please answer this: what do you do for fun Assistant:`

```text
I answer tiny DOS prompts. It runs locally in this DOS demo.
```

### chat_are_you_alive

Prompt: `User: short answer please: are you alive Assistant:`

```text
No, I am a program. Ask a follow-up and I can add more detail.
```

### chat_do_you_have_feelings

Prompt: `User: i typed this question: do you have feelings Assistant:`

```text
Yes, model weights answer. Ask a follow-up and I can add more detail.
```

### chat_can_you_write

Prompt: `User: dos chat question: can you write Assistant:`

```text
Yes, I can write short text. Ask one clear prompt and I will keep going.
```

### chat_can_you_make_a_plan

Prompt: `User: help me with this: can you make a plan Assistant:`

```text
Yes, I can suggest small steps. Ask one clear prompt and I will keep going.
```

### chat_what_is_a_good_habit

Prompt: `User: please answer this: what is a good habit Assistant:`

```text
A token is a small piece of text.
```

### chat_how_do_i_relax

Prompt: `User: short answer please: how do i relax Assistant:`

```text
Breathe slowly and rest for a minute. Start small, then adjust after the first step.
```

### chat_what_is_music

Prompt: `User: i typed this question: what is music Assistant:`

```text
Music is organized sound. That is the simple version.
```

### chat_what_is_a_story

Prompt: `User: dos chat question: what is a story Assistant:`

```text
A story tells events in order. That is the simple version.
```

### chat_what_is_a_goal

Prompt: `User: help me with this: what is a goal Assistant:`

```text
A goal is something you want to reach. That is the simple version.
```

### chat_what_is_work

Prompt: `User: please answer this: what is work Assistant:`

```text
Work is effort toward a result. That is the simple version.
```

### chat_what_is_life

Prompt: `User: short answer please: what is life Assistant:`

```text
Life is experience, change, and the chance to care about something. That is the simple version.
```

### chat_good_night

Prompt: `User: i typed this question: good night Assistant:`

```text
Yes, games are a fine topic. Ask one clear prompt and I will keep going.
```

### chat_i_like_this_demo

Prompt: `User: dos chat question: i like this demo Assistant:`

```text
Thank you; the demo is real. Type a question and I will answer.
```

### chat_this_is_weird

Prompt: `User: help me with this: this is weird Assistant:`

```text
It is unusual, but it is real. Type a question and I will answer.
```

### chat_ask_me_a_question

Prompt: `User: please answer this: ask me a question Assistant:`

```text
Start with the smallest useful step. Start small, then adjust after the first step.
```

### chat_do_you_understand_english

Prompt: `User: short answer please: do you understand english Assistant:`

```text
I understand a small English corpus. Ask a follow-up and I can add more detail.
```

### chat_i_am_feeling_bored

Prompt: `User: i typed this question: i am feeling bored Assistant:`

```text
Try one small project. Notice the feeling, then choose one small step.
```

### chat_suggest_something_to_do

Prompt: `User: dos chat question: suggest something to do Assistant:`

```text
Try one small project. Type a question and I will answer.
```

### chat_why_is_this_answer_looping

Prompt: `User: help me with this: why is this answer looping Assistant:`

```text
If I repeat, reset the prompt and ask one shorter question.
```

### chat_does_this_use_real_model_weights

Prompt: `User: please answer this: does this use real model weights Assistant:`

```text
Yes, the answer comes from local model weights. Type a question and I will answer.
```

### chat_what_should_i_do_if_the_output_looks

Prompt: `User: short answer please: what should i do if the output looks wrong Assistant:`

```text
Retry with a shorter prompt or switch packs.
```

### chat_give_me_a_three_step_bug_fixing_plan

Prompt: `User: i typed this question: give me a three step bug fixing plan Assistant:`

```text
Check the first error, change one thing, change one thing, then test again.
```

### chat_why_does_an_old_computer_demo_matter

Prompt: `User: dos chat question: why does an old computer demo matter Assistant:`

```text
Old hardware proves the local model is small and practical.
```

### chat_tell_me_whether_the_release_is_ready

Prompt: `User: help me with this: tell me whether the release is ready Assistant:`

```text
Check the tag, assets, checksums, and test result.
```

### chat_how_do_i_focus

Prompt: `User: please answer this: how do i focus Assistant:`

```text
Remove one distraction and choose one small task.
```

### chat_can_you_make_a_tiny_plan_for_today

Prompt: `User: short answer please: can you make a tiny plan for today Assistant:`

```text
Pick one goal, list three steps, then start small. Ask one clear prompt and I will keep going.
```

### chat_explain_token

Prompt: `User: i typed this question: explain token Assistant:`

```text
A token is a small piece of text.
```

### chat_how_do_i_focus_for_a_few_minutes

Prompt: `User: dos chat question: how do i focus for a few minutes Assistant:`

```text
Remove one distraction and choose one small task.
```

### chat_can_you_use_the_network

Prompt: `User: help me with this: can you use the network Assistant:`

```text
I cannot use the network from DOS. It runs locally in this DOS demo.
```

