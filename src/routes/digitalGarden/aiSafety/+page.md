# Limitations on Safe, Trusted, Artificial General Intelligence

[Paper link](https://arxiv.org/pdf/2509.21654)

From the abstract:

> We define safety of a system as the property that it never makes any false claims, trust as
> the assumption that the system is safe, and AGI as the property of an AI system always matching or
> exceeding human capability. Our core finding is that—for our formal definitions of these notions—a safe
> and trusted AI system cannot be an AGI system: for such a safe, trusted system there are task instances
> which are easily and provably solvable by a human but not by the system.

Why do they say "easily and provably solved by a human"? What does easily mean here?

> Theorem 1.5. If an AI system is safe and trusted, then it cannot be an AGI system. In particular, it is not
> an AGI system for the tasks of program verification, planning and determining graph reachability.

![Goedel Program](/images/digitalGarden/goedelProgram.png "Goedel Program")

> Somewhat more speculatively, note that our constructions rely on self-referential calls to the AI system,
> and when systems have general-purpose capabilities, such calls may not be implausible.

> However, our goal is not to argue for strict superiority of human reasoning over AI, but to show a separation: for
> safe, trusted AI systems there are instances that humans can solve, but which are not solvable by the system.

Ok but other AIs can also solve these problems. They say nothing special about the humans or AIs, only that agents that are safe and trusted can't solve these self-referential problems, and any competent agent that isn't the agent being referred to can solve it.

Could this limitation be used anywhere? When would we even care about safety on a self-referential problem? When would self-referential problems even show up? Maybe API calls and coding with the AI itself? Issues with planning graphs? But would these kinds of weird halting like algorithms show up practically?
