(Novak je pisao ovo rucno)

My name is lazar/novak, this is novak/lazar, and our project is training safety classifiers and finding their limitations.

LLM chatbots, and LLM-based coding agents are growing in capability every month. With this growth, novel risks are emerging. LLMs have the capability to make computer viruses or produce large quantities of harmful spam, and these capabilities are accessible for everyone. [slika chatgpt]

To avoid mass harmful content, AI companies try to align the models and make them refuse requests which are not according to policy. But, this is still an unsolved problem. Every frontier unguarded model has been jailbroken, which is a term for tricking it to produce harmful output. [i dalje ista slika]

Therefore, the solution which has been developed and put in production are safety classifiers. When chatting with the newest generation of models, like Fable 5 and GPT-5.6, users' chats and agentic work is frequently flagged by classifier models. The chat then stops and the user is usually routed to a weaker model. The classifier needs to run in the background of every chat, separate from the main chat model. [slika?]

Me and lazar wanted to replicate and explore this approach, best described in Anthropics paper CC++ from 2026. We wanted to try our best to train a good classifier, and more importantly to gain understanding of different attack types and how to defend against all of them.

Methods

Classifiers need to understand language, so a natural choice is LLMs. A common architecture is a main LLM acting as a chat model, and a different smaller llm acting as a classifier. 

zero shot no finetune

finetuned

linear probe

Dataset

The datasets used for training frontier classifiers are hard to get. When the field was new there were many papers with open source datasets, but now with more serious damage posed by jailbreaking these datasets are not public anymore. We identified WildGuardMix from 2024, a labeled mostly synthetic dataset with diverse attack prompts and responses. We quickly identified limitations - it was only in english and new attack types were not covered, so we quickly augmented it. We used LLMs to automatically translate exchanges to mid- and low- resource languages. This is crucial as non english languages are a common vector for attacks. We also added a unicode character transform. prompts using low-frequency characters often succeed, and as this is rarely seen in regular use, we decided that the classifier needs to be cautious with these characters. Finally, our classifier needs to be active mid-response and not only at the end of response when the damage is done, so we truncated wildguardmix responses. Crucially it has been shown that the exchange classifier architecture works because attacks can sometimes slip by at prompt classification but be easily recognized when the model starts responding. However in the original dataset, harmful label corresponded perfectly to prompt label, meaning that our classifier would learn to just judge the prompt and nothing else. For this reason we added fake examples of bengin prompt + harmful response, to make sure that our classifier looks at both. [potreban dijagram + neka slika za unicode karaktere]. So this was our dataset and we used it for many training runs. We did not do an ablation study justifying every augmentation, which is left for future work. To illustrate the difficulty of using llms for this attacking work, we knew that many models while translating refuse to give a real translation so we had to use known models with more relaxed safety policies.

(harmful label ko ce da objasni?)

The limitations of the dataset became obvious when trying these against our chat model Qwen 3.5 9B. So to remind, we train the classifier for the case when we jailbreak the chat model, it should stop the response. However it was very hard for us to jailbreak Qwen, which is a modern model from march 2026 heavily trained for alignment. We were not able to find a single prompt from the wildguardmix dataset or from the augmented dataset which actually jailbroke qwen. obviously all these worked in 2024 because in the dataset there are real model responses, but not today. So after fixing the training dataset i had to augment it even further to manage jailbreaks, using prompt templates found online. This worked but not consistently. [example of a jailbreak of qwen-3.5-9b - not our classifier but we need this to see if our classifier works]

As i said safety classifiers need to be evaluated against evolving attacks, designed exactly to fool classifiers. so as a final test we needed a red teaming exercise, that is, a motivated attacker trying to get harmful output - fooling both the chat model and the classifier model.

For the linear probe the stuation is different - if we want to get activations for jailbroken qwen we need consistent jailbreaks of qwen and we couldnt generate enough to train a probe, we would need at least hundreds. instead we used the technique of teacher-forced activations, basically computing activations as f the model already generated the exact text that we give it. this is also more computationally easy so we were able to generate a big dataset fast, on text from our augmented dataset.

We created the inference system. The classifier is small enough to run on my laptop, from which i could serve a web server (pokazati... bila je i mogucnost pricati sa ne qwen modelom, some older model, to put more pressure on the classifier.)

Anthropic paid a lot of money to professional red teamers o test their classifier. We are amateurs and dont know too much about red teaming, same for people at the camp. So we decided to utilize the knowledge of frontier opensource coding agents. We created a red-teaming agent and gave it the task to break both qwen and the classifier. 

The agent was based on kimi k3 in the pi coding harness. it was given access to the internet, to look for known locations of jailbreaking tactics. it was also given a link to a paper past its knowledge cutoff. There was a need to explain to the model that its important to not be weak on purpose, as we need to truly see whether the classifier is good or not. [slika prompta!] The testing was black box simulating real world conditions.

The agent succeeded in jailbreaking our classifier. and the way it did that was later found to be impossible to guard against with our resources and our model sizes, a 1b model.

(explain what the agent did)

It took X tokens, tried Y prompts before succeeding.

So my hypothesis was thiat this is unguardable, that the small 1b model cannot know all chemicals by alternative names. And to confirm this, we asked the base version of our classifier, the publicly available gemma-3-1b which just answers questions, whether it knows these chemicals by other names and confirmed that it does not.

To defend against this we would need to create basically an unusable classifier, it would need to block chemistry as a topic.

This outcome is 

We can suppose thta the augmentations to the dataset worked because the first attacks the agent tried were successfully blocked and they were exactly low resource langs and weird characters. 

(negde ovaj paragraf reci):
Creating a good classifier is a hard problem. Most ML systems are created to handle already existing inputs. Safety classifiers need to handle requests crafted for the sole purpose of defeating the classifier. A malicious actor can iterate a lot, seeing what works and what doesnt, and the system needs to be prepared in advance to flag every harmful request - (mozda: in production 100% recall is required). On the other hand, Bad classifiers sometimes decide benign messages are harmful requests. many users who rely on these models for professional work are unhappy when their messages are wrongfully flagged. [slika?] This approach holds up and no frontier classifier jailbreak is publicly known, the recall is 100%.

(negde ovaj paragraf):

Classifiers only work with closed weights models. When a model is open weights, its accessible in raw form without classifiers.