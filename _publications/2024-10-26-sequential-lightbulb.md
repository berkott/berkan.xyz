---
title: "A Sequential Lightbulb Problem"
collection: publications
permalink: /publication/2024-10-26-sequential-lightbulb
excerpt: 'This paper studies an online version of the light bulb problem and gives simple lower-bounds and develops algorithms that allude to a space-runtime tradeoff.'
date: 2024-10-26
venue: 'Presented at the Fall Fourier Talks, University of Maryland'
paperurl: 'https://berkan.xyz/files/lightbulb.pdf'
citation: "Noah Bergam, Berkan Ottlik, Arman Özcan. (2024). &quot;A Sequential Lightbulb Problem&quot;."
---
The light bulb problem is a fundamental unsupervised learning problem about identifying correlations amidst noise. In this report, we explore an online formulation of the light bulb problem. In light of the fact that the known offline approaches to the problem require super-linear space, we develop a simple algorithm which can solve the online problem using \(O(n)\) space in \(\tilde O(n)\) rounds. We then provide an enhanced algorithm which can toggle a tradeoff between space and rounds: namely, for any \(\alpha\in (0,1)\), one can solve the online lightbulb problem with \(O(n^{1-\alpha})\) space and \(\tilde O(n^{1+\alpha})\) rounds. This method can be extended to allow for constant space, at the expense of quadratic rounds. Finally, we prove relevant lower bounds for the problem.

<!-- [Download paper here](https://www.nctatechnicalpapers.com/Paper/2021/2021-machine-learning-and-proactive-network-maintenance-transforming-today-s-plant-operations/download)

Recommended citation: Your Name, You. (2015). "Paper Title Number 3." <i>Journal 1</i>. 1(3). -->