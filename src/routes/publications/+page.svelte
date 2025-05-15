<script lang="ts">
  import PaperDescription from '$lib/paperDescription.svelte';
  import { Math } from "svelte-math";
</script>

<p class="mb-4">
Updated May 10, 2025. See my <a href="https://scholar.google.com/citations?user=cAkhZCgAAAAJ">Google Scholar profile</a> for the most up-to-date record of my publications.
</p>

<PaperDescription
  title="Gradient Flow Dynamics of Teacher-Student Distillation with the Squared Loss"
  link="https://berkan.xyz/files/underparameterizedDynamics.pdf"
  authors="Berkan Ottlik"
  year="2024"
  venue="Presented at the Summer@Simons poster session"
  rawBibtex={
`@article{ottlik2024gradient,
  author = "Ottlik, Berkan",
  title = "Gradient Flow Dynamics of Teacher-Student Distillation with the Squared Loss",
  journal = "Presented at the Summer@Simons poster session",
  year = "2024",
  url = "https://berkan.xyz/files/underparameterizedDynamics.pdf"
}`
}
>
  <p>
    We study a teacher-student learning setup, where a "student" one layer neural network tries to approximate a fixed "teacher" one layer neural network. We analyze the population gradient flow dynamics in the previously unstudied setting with exactly and under-parameterization, even Hermite polynomial activation functions, and squared loss. In the toy model with 2 teacher neurons and 2 student neurons, we fully characterize all critical points. We identify "tight-balance" critical points which are frequently encountered in simulation and greatly slow down training. We prove that with favorable initialization, we avoid tight-balance critical points and converge to the global optimum. We extend tight-balance critical points and favorable initializations to the multi-neuron exact and under-parameterized regimes. Additionally, we compare dynamics under the squared loss to the simpler correlation loss and describe the loss landscape in the multi-neuron exact and under-parameterized regimes. Finally, we discuss potential implications our work could have for training neural networks with even activation functions.
  </p>
</PaperDescription>

<PaperDescription
  title="A Sequential Lightbulb Problem"
  link="https://berkan.xyz/files/lightbulb.pdf"
  authors="Noah Bergam, Berkan Ottlik, Arman Özcan"
  year="2024"
  venue="Presented at the Fall Fourier Talks, University of Maryland"
  rawBibtex={
`@article{bergam2024sequential,
  author = "Bergam, Noah and Ottlik, Berkan and Özcan, Arman",
  title = "A Sequential Lightbulb Problem",
  journal = "Presented at the Fall Fourier Talks, University of Maryland",
  year = "2024",
  url = "https://berkan.xyz/files/lightbulb.pdf"
}`
}
>
  <p>
    The light bulb problem is a fundamental unsupervised learning problem about identifying correlations amidst noise. In this report, we explore an online formulation of the light bulb problem. In light of the fact that the known offline approaches to the problem require super-linear space, we develop a simple algorithm which can solve the online problem using <Math latex={String.raw`O(n)`}/> space in <Math latex={String.raw`\tilde{O}(n)`}/> rounds. We then provide an enhanced algorithm which can toggle a tradeoff between space and rounds: namely, for any <Math latex={String.raw`\alpha \in (0,1)`}/>, one can solve the online lightbulb problem with <Math latex={String.raw`O(n^{1-\alpha})`}/> space and <Math latex={String.raw`\tilde{O}(n^{1+\alpha})`}/> rounds. This method can be extended to allow for constant space, at the expense of quadratic rounds. Finally, we prove relevant lower bounds for the problem.
  </p>
</PaperDescription>

<PaperDescription
  title="The Effect of Model Capacity on the Emergence of In-Context Learning"
  link="https://openreview.net/pdf?id=YZM9g0Mi9a"
  authors="Berkan Ottlik, Narutatsu Ri, Daniel Hsu, Clayton Sanford"
  year="2024"
  venue="ICLR 2024 Workshop on Understanding of Foundation Models (ME-FoMo)"
  rawBibtex={
`@inproceedings{ottlik2024effect,
  author = "Ottlik, Berkan and Ri, Narutatsu and Hsu, Daniel and Sanford, Clayton",
  title = "The Effect of Model Capacity on the Emergence of In-Context Learning",
  booktitle = "ICLR 2024 Workshop on Understanding of Foundation Models (ME-FoMo)",
  year = "2024",
  url = "https://openreview.net/pdf?id=YZM9g0Mi9a"
}`
}
>
  <p>
    This paper investigates the relationship between model capacity and the emergence of in-context learning under a simplified statistical framework in the transformer model. When model capacity is restricted enough, transformers shift from learning the Bayes optimal estimator for the training task distribution to an estimator that is suitable for out-of-distribution tasks. This shift is attributed to the restricted model's inability to fully memorize the training task distribution. Further experiments examine how the transformer's hyper-parameters impact its capacity for memorization.
  </p>
</PaperDescription>

<PaperDescription
  title="Machine Learning and Proactive Network Maintenance: Transforming Today's Plant Operations"
  link="https://www.nctatechnicalpapers.com/Paper/2021/2021-machine-learning-and-proactive-network-maintenance-transforming-today-s-plant-operations/"
  authors="Berkan Ottlik, Brady Volpe"
  year="2021"
  venue="Fall Technical Forum: SCTE, NCTA, CableLabs"
  rawBibtex={
`@article{ottlik2021machine,
  author = "Ottlik, Berkan and Volpe, Brady",
  title = "Machine Learning and Proactive Network Maintenance: Transforming Today's Plant Operations",
  journal = "Fall Technical Forum: SCTE, NCTA, CableLabs",
  year = "2021",
  url = "https://www.nctatechnicalpapers.com/Paper/2021/2021-machine-learning-and-proactive-network-maintenance-transforming-today-s-plant-operations/"
}`
}
>
  <p>
    We first describe the type of RF impairments observable by PNM. We apply unsupervised learning methods to detect these impairments. We integrate our approach into CableLab's spectral impairment detector (SID) to substantially improve on SID's impairment classifiers. Finally, we consider how feedback from end users could enable a supervised learning approach.
  </p>
</PaperDescription>