<script>
  import ProjectDescription from '$lib/projectDescription.svelte';
</script>

<p>Here are some of my projects. You can find everything on my <a href="https://github.com/berkott">GitHub</a>.</p>

<ProjectDescription 
  title="Gardening Handbook"
  link="files/gardeningHandbook.pdf"
  date="May 2025"
  description="I documented everything I learned from Professor <a href='https://www.engineering.columbia.edu/faculty-staff/directory/j-thomas-tommy-vaughan-jr'>Thomas Vaughan</a> and other gardeners at the Columbia University Victory Garden throughout undergrad."
  imageSrc="/images/projects/garden.jpg"
/>

<ProjectDescription 
  title="Decoding Behavior from Neural Data"
  link="files/neuralStats.pdf"
  date="December 2023"
  description="Designed new and implemented existing algorithms for decoding mouse wheel speed behavior from neuron-level brain recordings."
  imageSrc="/images/projects/neuralStats.png"
/>

<ProjectDescription 
  title="LLM Trajectory Analysis"
  link="files/llmTrajectories.pdf"
  date="March 2023"
  description="Analyzed trajectories of language model embeddings projected onto a 3-sphere using UMAP. Won the special Nomic bounty at the NYU Generative AI Betaworks Hackathon."
  imageSrc="/images/projects/llmTrajectories.png"
/>

<ProjectDescription 
  title="LangShell"
  link="https://devpost.com/software/langshell"
  date="February 2023"
  description="Our shell tool, called LangShell, is a GPT-3 based shell assistant with a memory. You can tell LangShell facts about yourself, and you will never forget them (unless you want it to). We built LangShell using Python, the OpenAI GPT-3 API, and the Huggingface MiniLM-L6-v sentence similarity API. It is based on a fork of Shell GPT, and we plan to release it as a Pull request on the original repo."
  imageSrc="/images/projects/langShell.png"
/>

<ProjectDescription 
  title="Charizzma"
  link="https://devpost.com/software/charizzma"
  date="February 2023"
  description="Charizzma is an AI conversational advice-giver that helps you respond smoothly when you need it most. Specifically, we used real-time transcription, keyword extraction, and speech synthesis to deliver advice in real time."
  imageSrc="/images/projects/charizzma.png"
/>

<ProjectDescription 
  title="Unsupervised Disentanglement"
  link="https://github.com/berkott/disentanglementAndScale"
  date="December 2022"
  description="Conducted a literature review of unsupervised disentanglement and ran experiments with disentanglement and scale. Here are the <a href='files/unsupervisedDisentanglementLiteratureReviewAndExplorationWithScale.pdf'>report</a> and <a href='files/unsupervisedDisentanglementMathNotes.pdf'>math notes</a>."
  imageSrc="/images/projects/disentanglement.png"
/>

<ProjectDescription 
  title="Vision Transformer Interpretability"
  link="https://github.com/berkott/cvInterp"
  date="August 2022"
  description="Interpreted a vision transformer. I <a href='https://github.com/berkott/lucent'>made changes</a> to the <a href='https://github.com/greentfrapp/lucent'>lucent library</a>, created a <a href='https://github.com/berkott/visFrontend'>frontend to visualize things</a>, and made a <a href='https://docs.google.com/presentation/d/1b7RT35xOf--fcfwsDPTmoeHCBLKEGBfJq69bBKrdwec/edit?usp=sharing'>slideshow write-up</a> of my work."
  imageSrc="/images/projects/visionTransformerInterp.png"
/>

<ProjectDescription 
  title="Pong Bot"
  link="https://github.com/berkott/pong-bot"
  date="December 2021"
  description="Created a pong playing robot for Columbia Art of Engineering project."
  imageSrc="/images/projects/pongBot.png"
/>

<ProjectDescription 
  title="Covid Detection Kaggle"
  link="https://github.com/berkott/covid-detection-kaggle"
  date="August 2020"
  description="Created a model to view lung X-rays to localize lung opacities and use them to make COVID-19 diagnoses."
  imageSrc="/images/projects/xray.png"
/>

<ProjectDescription 
  title="Dev Fruit"
  link="https://www.youtube.com/channel/UC79P_zoIzbrdxlNy1nVG9mQ"
  date="May 2020"
  description="Created fun fruit themed videos of cool projects."
  imageSrc="/images/projects/logoOrangeV1.png"
/>

<ProjectDescription 
  title="Blind Assist"
  link="https://github.com/berkott/NavigationAssistant"
  date="March 2020"
  description="Created a headset and program to be used by visually impaired pedestrians."
  imageSrc="/images/projects/blind.png"
/>

<ProjectDescription 
  title="Mole Project"
  link="https://github.com/berkott/mole"
  date="October 2018"
  description="Created a robotic mole plush toy that talks and takes voice commands for mole day."
  imageSrc="/images/projects/mole.png"
/>