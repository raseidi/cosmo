# related papers

[OO prediction based on DECLARE LTLf constraints (EAAI 2023)](https://www.sciencedirect.com/science/article/pii/S0952197623010837#tbl20)

- prescribes to users temporal relations among activities that have to be preserved or violated in order to achieve a desired outcome.
- the prescription and recommendation are defined as a tuple $r=(tc, c)$, where $tc$ is the temporal constraint (e.g. $RESPONSE(a,b)$) and $c \in C$, where $C$ is the set of conditions (status would fit better this definition) $C={satisfied, validated}$.

[Predictive Monitoring of Business Processes (CAISE 2014)]()

- if a given data snapshot leads to a business constraint fulfillment and with what probability
  - a data snapshot is an ongoing trace plus the same attributes from the   most similar (previously known) prefix (a.k.a. control-flow match)
  - decision tree


# paper structure

1. Introduction
2. Related works
3. Problem definition and motivation
4. Proposal
5. Experiments
6. Conclusion

# simulations' findings

## SEPSIS

### Existence 

- existence template works well only for IV Antibiotics
- however, i do not understand why IV Anti works well but Admission NC does not. A possible reason:
  - IV Anti appears only once in cases
  - wheres Adm NC might appear several times (indicating loops)
- for airbitrary sets of constraints (acutally only the ones that work well, 15 in total), we run the generation procedure 20x.
  - this results in 15*20=300 cases
  - from the 300 cases, 276 properly satisfied the imposed condition
  - 97 variants were generated
- **WE ARE GOING TO IGNORE EXISTENCE FOR VISUALIZATION** in the paper

### Positive relations

- if A, B is next
- in the original log, triage has a well distributed branch, i.e. it might be followed by different activities e.g. iv, crp, leucocytes.
- We took this relation and forbid the transition triage -> crp
- it worked perfectly and triage is only going to lactic and leococytes
- the opposite does not work (we have to first change our check_rule function)

### Exclusive CHOICE

- A or B but not both
- ours: Triage OR LacticAcid at least once but not both
- from the original log, there is a well distributed branch probability from TRIAGE: for the 900 cases, there are 200 branches towards 4 different activities
- in the simulated log, although we do not satisfy the constraint 100%, there is a signifcant change in the branch probabilities: the B activity is very less likely to happen after triage


# Visualization apromore

- original PP: 10%
- choice 60% arcs
- PR 100%

# background

- instead of prefixes, we use sequences. The reason for that is due to the declare rules extraction. Each trace can be encoded as a new feature vector containing the declare rules.
- DECLARE templates can be gathered into four main groups
  - existence
    - the templates in this group have only one parameter and check either the number of its occurrences in a trace or its position in the trace.
  - choice
    - the templates in this group have two parameters and check if (at least) one of them occurs in a trace.
  - positive relation
    - the templates in this group have two parameters and check the relative position between the two corresponding activities
  - negative relation
    - the templates in this group have two parameters and check that the two corresponding activities do not occur together or do not occur in a certain order.
- for ongoing traces, these constraints can be (possibly) satisfied/violated;
  - in our contribution we train with satisfied/violated only but we test with possibly, i.e. we simulate changes during the trace execution

## LTLf

- X_f: `f` has to hold in the next position of a sequence.
- G_f: `f` has to hold always (Globally) in the subsequent positions of a sequence.
- F_f: `f` has to hold eventually (in the Future) in the subsequent position of a sequence.
f U g: `f` has to hold in a sequence at least Until `g` holds. `g` must hold in the current or in a future position.

Examples 

- if CHAIN RESPONSE(a, b) is valid, ALTERNATE RESPONSE (a, b) is as well.
  - CR(a,b): ..., a, b, ...
  - AR(a,b): ...,a,...c,...,b,... (but a second a is not allowed between the first a and next b)
  - CR(a,b): ...,a,a,b,...
  - AR(a,b): ...,a,[a,b],... ????: this is true because if CR is valid AR is too


# takes

- the approach is naive but it works
- lots of work must be done to better manage the rules
- expert knowledge to guide the simulation is key