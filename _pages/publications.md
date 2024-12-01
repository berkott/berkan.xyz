---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% include base_path %}

<p>See my <a href="https://scholar.google.com/citations?user=cAkhZCgAAAAJ">Google Scholar profile</a> for the most up-to-date record of my publications.</p>

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
