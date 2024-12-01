---
title: "Gradient Flow Dynamics of Teacher-Student Distillation with the Squared Loss"
collection: publications
permalink: /publication/2024-11-20-underparameterized-dynamics
excerpt: 'This paper studies the gradient flow dynamics teacher-student distillation setup with squared loss and multiple student and teacher neurons.'
date: 2024-11-20
venue: 'Presented at the Summer@Simons poster session.'
paperurl: 'https://berkan.xyz/files/underparameterizedDynamics.pdf'
citation: "Berkan Ottlik. (2024). &quot;Gradient Flow Dynamics of Teacher-Student Distillation with the Squared Loss&quot;."
---
We study a teacher-student learning setup, where a "student" one layer neural network tries to approximate a fixed "teacher" one layer neural network. We analyze the population gradient flow dynamics in the previously unstudied setting with exactly and under-parameterization, even Hermite polynomial activation functions, and squared loss. In the toy model with 2 teacher neurons and 2 student neurons, we fully characterize all critical points. We identify "tight-balance" critical points which are frequently encountered in simulation and greatly slow down training. We prove that with favorable initialization, we avoid tight-balance critical points and converge to the global optimum. We extend tight-balance critical points and favorable initializations to the multi-neuron exact and under-parameterized regimes. Additionally, we compare dynamics under the squared loss to the simpler correlation loss and describe the loss landscape in the multi-neuron exact and under-parameterized regimes. Finally, we discuss potential implications our work could have for training neural networks with even activation functions.

<!-- [Download paper here](https://www.nctatechnicalpapers.com/Paper/2021/2021-machine-learning-and-proactive-network-maintenance-transforming-today-s-plant-operations/download)

Recommended citation: Your Name, You. (2015). "Paper Title Number 3." <i>Journal 1</i>. 1(3). -->