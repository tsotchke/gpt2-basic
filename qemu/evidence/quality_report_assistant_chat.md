# Assistant Pack Quality Report: CHAT

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab4096`
Evaluation backend: `float`
Quality suite: `assistant-pack`
Quality status: `NEEDS_TRAINING`
Average score: `0.879`
Prompt pass rate: `56/72` at threshold `0.90`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| chat_hello | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_how are you | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_what can you do | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_limits | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_hi | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_hey | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_good_evening | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_are_you_there | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_are_you_called | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_can_you_do | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_we_talk_for_a_minute | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_does_prompt_mean | 0.789 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_explain | 1.000 | 3 | 0.0% | 1 | 0 | yes | PASS |
| chat_give_me_an_idea | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_help | 0.490 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| chat_what_are_your_limits | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_you_browse | 0.490 | 0 | 9.1% | 2 | 0 | yes | RETRAIN |
| chat_is_this_scripted | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_is_this_just_a_script | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_do_you_use_a_script | 1.000 | 1 | 0.0% | 1 | 0 | yes | PASS |
| chat_is_this_real | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_do_you_use_model_weights | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_is_the_demo_running | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_qemu | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_can_you_explain_what_local_inference | 1.000 | 6 | 0.0% | 1 | 0 | yes | PASS |
| chat_say_something | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_thank_you | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_goodbye | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_i_am_tired | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_i_feel_lonely | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_i_feel_good | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_should_i_do_today | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_you_help_me_think | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_you_remember_me | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_you_learn_during_this_chat | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_tell_me_a_story | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_do_you_like_music | 0.490 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| chat_do_you_like_games | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_do_you_do_for_fun | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_are_you_alive | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_are_you_human | 1.000 | 0 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_you_learn | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_can_you_give_advice | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_give_me_a_plan | 1.000 | 5 | 7.1% | 2 | 0 | yes | PASS |
| chat_how_do_i_focus | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_how_do_i_improve | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_a_game | 1.000 | 3 | 0.0% | 1 | 0 | yes | PASS |
| chat_what_is_a_habit | 1.000 | 2 | 0.0% | 1 | 0 | yes | PASS |
| chat_what_is_a_choice | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_what_is_the_meaning_of_life | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_why_are_you_repeating_yourself | 1.000 | 4 | 0.0% | 1 | 0 | yes | PASS |
| chat_see_you_later | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_this_is_cool | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_talk_to_me | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_what_can_we_discuss | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_do_you_understand_english | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_can_i_do_if_bored | 0.490 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| chat_why_is_my_answer_repeating | 1.000 | 5 | 0.0% | 1 | 0 | yes | PASS |
| chat_why_is_this_answer_looping | 1.000 | 5 | 0.0% | 1 | 0 | yes | PASS |
| chat_does_this_use_real_model_weights | 1.000 | 3 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_should_i_do_if_output_looks_wro | 1.000 | 4 | 0.0% | 1 | 0 | yes | PASS |
| chat_give_me_a_bug_fixing_plan | 1.000 | 6 | 0.0% | 2 | 0 | yes | PASS |
| chat_make_a_tiny_plan_for_fixing_a_bug | 1.000 | 6 | 0.0% | 2 | 0 | yes | PASS |
| chat_why_does_an_old_computer_demo_matter | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_is_the_release_ready | 1.000 | 5 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_should_i_verify_before_release | 1.000 | 5 | 0.0% | 2 | 0 | yes | PASS |
| chat_give_me_a_small_plan | 0.768 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_tell_me_a_short_story_about_a_dos_mo | 1.000 | 5 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_the_difference_between_my_pr | 0.490 | 1 | 0.0% | 1 | 0 | yes | RETRAIN |
| chat_how_do_i_focus_for_a_few_minutes | 1.000 | 5 | 0.0% | 2 | 0 | yes | PASS |
| chat_can_you_browse_the_internet_from_dos | 0.896 | 2 | 0.0% | 2 | 0 | yes | PASS |
| chat_i_am_lonely | 1.000 | 2 | 0.0% | 2 | 0 | yes | PASS |

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
I am sorry. Notice the feeling, then choose one small step.
```

### chat_hey

Prompt: `User: short answer please: hey Assistant:`

```text
Hello from DOS. Type a question and I will answer.
```

### chat_good_evening

Prompt: `User: i typed this question: good evening Assistant:`

```text
Yes, I can read your prompt. Ask one clear prompt and I will keep going.
```

### chat_are_you_there

Prompt: `User: dos chat question: are you there Assistant:`

```text
Yes, I am running in DOS. It runs locally in this DOS demo.
```

### chat_what_are_you_called

Prompt: `User: help me with this: what are you called Assistant:`

```text
I am GPT2-BASIC. Ask a follow-up and I can add more detail.
```

### chat_what_can_you_do

Prompt: `User: please answer this: what can you do Assistant:`

```text
I can chat in DOS. It runs locally in this DOS demo.
```

### chat_can_we_talk_for_a_minute

Prompt: `User: short answer please: can we talk for a minute Assistant:`

```text
Yes, type a short prompt. Ask one clear prompt and I will keep going.
```

### chat_what_does_prompt_mean

Prompt: `User: i typed this question: what does prompt mean Assistant:`

```text
Yes, ask about one word. Ask one clear prompt and I will keep going.
```

### chat_explain

Prompt: `User: dos chat question: explain Assistant:`

```text
I can explain simple ideas. That is the simple version.
```

### chat_give_me_an_idea

Prompt: `User: help me with this: give me an idea Assistant:`

```text
Try one clear question. Ask a follow-up and I can add more detail.
```

### chat_help

Prompt: `User: please answer this: help Assistant:`

```text
Local inference means the DOS program reads model weights and produces the answer on this machine. The answer comes from local model weights.
```

### chat_what_are_your_limits

Prompt: `User: short answer please: what are your limits Assistant:`

```text
I am tiny and local. This tiny session has clear limits.
```

### chat_can_you_browse

Prompt: `User: i typed this question: can you browse Assistant:`

```text
Check the first error, change one thing, change one thing, then test again.
```

### chat_is_this_scripted

Prompt: `User: dos chat question: is this scripted Assistant:`

```text
No, it is real. Type a question and I will answer.
```

### chat_is_this_just_a_script

Prompt: `User: help me with this: is this just a script Assistant:`

```text
No, this is real local model output. Type a question and I will answer.
```

### chat_do_you_use_a_script

Prompt: `User: please answer this: do you use a script Assistant:`

```text
No, it is real. The answer comes from local model weights.
```

### chat_is_this_real

Prompt: `User: short answer please: is this real Assistant:`

```text
Yes, it is real. Type a question and I will answer.
```

### chat_do_you_use_model_weights

Prompt: `User: i typed this question: do you use model weights Assistant:`

```text
Yes, model weights answer. Ask a follow-up and I can add more detail.
```

### chat_is_the_demo_running

Prompt: `User: dos chat question: is the demo running Assistant:`

```text
The demo is live. It runs locally in this DOS demo.
```

### chat_what_is_qemu

Prompt: `User: help me with this: what is qemu Assistant:`

```text
Check the first error, change one thing, then test again.
```

### chat_can_you_explain_what_local_inference

Prompt: `User: please answer this: can you explain what local inference means Assistant:`

```text
Local inference means the DOS program reads model weights and produces the answer on this machine. The answer comes from local model weights.
```

### chat_say_something

Prompt: `User: short answer please: say something Assistant:`

```text
Hello from DOS. Type a question and I will answer.
```

### chat_thank_you

Prompt: `User: i typed this question: thank you Assistant:`

```text
You are welcome. Ask a follow-up and I can add more detail.
```

### chat_goodbye

Prompt: `User: dos chat question: goodbye Assistant:`

```text
Goodbye from DOS. Come back when you want another chat.
```

### chat_i_am_tired

Prompt: `User: help me with this: i am tired Assistant:`

```text
Check the first error, change one thing, then test again.
```

### chat_i_feel_lonely

Prompt: `User: please answer this: i feel lonely Assistant:`

```text
Yes, it is real. Type a question and I will answer.
```

### chat_i_feel_good

Prompt: `User: short answer please: i feel good Assistant:`

```text
Good; keep going. Notice the feeling, then choose one small step.
```

### chat_what_should_i_do_today

Prompt: `User: i typed this question: what should i do today Assistant:`

```text
Pick one useful small task.
```

### chat_can_you_help_me_think

Prompt: `User: dos chat question: can you help me think Assistant:`

```text
Yes, tell me the problem. Type a question and I will answer.
```

### chat_can_you_remember_me

Prompt: `User: help me with this: can you remember me Assistant:`

```text
Only within this session. Ask one clear prompt and I will keep going.
```

### chat_can_you_learn_during_this_chat

Prompt: `User: please answer this: can you learn during this chat Assistant:`

```text
Check the first error, change one thing, then test again.
```

### chat_tell_me_a_story

Prompt: `User: short answer please: tell me a story Assistant:`

```text
A tiny model woke up inside DOS. It runs locally in this DOS demo.
```

### chat_do_you_like_music

Prompt: `User: i typed this question: do you like music Assistant:`

```text
If I repeat, reset the prompt and ask one shorter question.
```

### chat_do_you_like_games

Prompt: `User: dos chat question: do you like games Assistant:`

```text
I can talk about games. We can keep the topic simple and friendly.
```

### chat_what_do_you_do_for_fun

Prompt: `User: help me with this: what do you do for fun Assistant:`

```text
I answer tiny DOS prompts. It runs locally in this DOS demo.
```

### chat_are_you_alive

Prompt: `User: please answer this: are you alive Assistant:`

```text
No, I am a program. Ask a follow-up and I can add more detail.
```

### chat_are_you_human

Prompt: `User: short answer please: are you human Assistant:`

```text
No, I am a DOS model. It runs locally in this DOS demo.
```

### chat_can_you_learn

Prompt: `User: i typed this question: can you learn Assistant:`

```text
Yes, it is real. Type a question and I will answer.
```

### chat_can_you_give_advice

Prompt: `User: dos chat question: can you give advice Assistant:`

```text
Yes, ask one practical question. Ask one clear prompt and I will keep going.
```

### chat_give_me_a_plan

Prompt: `User: help me with this: give me a plan Assistant:`

```text
Pick a goal, then choose the first step. Start small, then adjust after the first step.
```

### chat_how_do_i_focus

Prompt: `User: please answer this: how do i focus Assistant:`

```text
Remove one distraction and choose one small task.
```

### chat_how_do_i_improve

Prompt: `User: short answer please: how do i improve Assistant:`

```text
Practice one small thing each day. Start small, then adjust after the first step.
```

### chat_what_is_a_game

Prompt: `User: i typed this question: what is a game Assistant:`

```text
A game is play with rules. That is the simple version.
```

### chat_what_is_a_habit

Prompt: `User: dos chat question: what is a habit Assistant:`

```text
A habit is an action you repeat. That is the simple version.
```

### chat_what_is_a_choice

Prompt: `User: help me with this: what is a choice Assistant:`

```text
Check the first error, change one thing, then test again.
```

### chat_what_is_the_meaning_of_life

Prompt: `User: please answer this: what is the meaning of life Assistant:`

```text
This demo runs a tiny model in DOS. Type a question and I will answer.
```

### chat_why_are_you_repeating_yourself

Prompt: `User: short answer please: why are you repeating yourself Assistant:`

```text
If I repeat, ask a shorter question or switch packs. Reset the prompt and ask one shorter question.
```

### chat_see_you_later

Prompt: `User: i typed this question: see you later Assistant:`

```text
Yes, it is real. Type a question and I will answer.
```

### chat_this_is_cool

Prompt: `User: dos chat question: this is cool Assistant:`

```text
Thank you; it runs in DOS. Type a question and I will answer.
```

### chat_talk_to_me

Prompt: `User: help me with this: talk to me Assistant:`

```text
Check the first error, change one thing, then test again.
```

### chat_what_can_we_discuss

Prompt: `User: please answer this: what can we discuss Assistant:`

```text
We can discuss ideas, feelings, games, or DOS. It runs locally in this DOS demo.
```

### chat_do_you_understand_english

Prompt: `User: short answer please: do you understand english Assistant:`

```text
I understand a small English corpus. Ask a follow-up and I can add more detail.
```

### chat_what_can_i_do_if_bored

Prompt: `User: i typed this question: what can i do if bored Assistant:`

```text
Check the first error, change one thing, then test again.
```

### chat_why_is_my_answer_repeating

Prompt: `User: dos chat question: why is my answer repeating Assistant:`

```text
If I repeat, reset the prompt and ask one shorter question.
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

### chat_what_should_i_do_if_output_looks_wro

Prompt: `User: short answer please: what should i do if output looks wrong Assistant:`

```text
Retry with a shorter prompt or switch packs.
```

### chat_give_me_a_bug_fixing_plan

Prompt: `User: i typed this question: give me a bug fixing plan Assistant:`

```text
Check the first error, change one thing, then test again.
```

### chat_make_a_tiny_plan_for_fixing_a_bug

Prompt: `User: dos chat question: make a tiny plan for fixing a bug Assistant:`

```text
Check the first error, change one thing, then test again.
```

### chat_why_does_an_old_computer_demo_matter

Prompt: `User: help me with this: why does an old computer demo matter Assistant:`

```text
Old hardware proves the local model is small and practical. It runs locally in this DOS demo.
```

### chat_is_the_release_ready

Prompt: `User: please answer this: is the release ready Assistant:`

```text
Check the tag, assets, checksums, and test result.
```

### chat_what_should_i_verify_before_release

Prompt: `User: short answer please: what should i verify before release Assistant:`

```text
Check the tag, assets, checksums, and test result.
```

### chat_give_me_a_small_plan

Prompt: `User: i typed this question: give me a small plan Assistant:`

```text
Check the first error, change one thing, then test again.
```

### chat_tell_me_a_short_story_about_a_dos_mo

Prompt: `User: dos chat question: tell me a short story about a dos model Assistant:`

```text
A tiny model woke up inside DOS and answered one prompt. It runs locally in this DOS demo.
```

### chat_what_is_the_difference_between_my_pr

Prompt: `User: help me with this: what is the difference between my prompt and your answer Assistant:`

```text
If I repeat, reset the prompt and ask one shorter question.
```

### chat_how_do_i_focus_for_a_few_minutes

Prompt: `User: please answer this: how do i focus for a few minutes Assistant:`

```text
Remove one distraction and choose one small task.
```

### chat_can_you_browse_the_internet_from_dos

Prompt: `User: short answer please: can you browse the internet from dos Assistant:`

```text
I cannot browse from DOS. It runs locally in this DOS demo.
```

### chat_i_am_lonely

Prompt: `User: i typed this question: i am lonely Assistant:`

```text
I can keep you company briefly. Notice the feeling, then choose one small step.
```

