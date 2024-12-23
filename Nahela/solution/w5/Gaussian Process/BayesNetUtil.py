
import networkx as nx

def tokenise_query(prob_query, verbose):
    if verbose: print("\nTOKENISING probabilistic query="+str(prob_query))

    query = {}
    prob_query = prob_query[2:]
    prob_query = prob_query[:len(prob_query)-1]
    query["query_var"] = prob_query.split("|")[0]
    query["evidence"] = prob_query.split("|")[1]

    evidence = {}
    if query["evidence"].find(','):
        for pair in query["evidence"].split(','):
            tokens = pair.split('=')
            evidence[tokens[0]] = tokens[1]
        query["evidence"] = evidence

    if verbose: print("query="+str(query))
    return query


# returns the parent of random variable 'child' given Bayes Net 'bn'
def get_parents(child, bn):
    for conditional in bn["structure"]:
        if conditional.startswith("P("+child+")"):
            return None
        elif conditional.startswith("P("+child+"|"):
            parents = conditional.split("|")[1]
            parents = parents[:len(parents)-1]
            return parents

    print("ERROR: Couldn't find parent(s) of variable "+str(child))
    exit(0)


# returns the probability of tuple V=v given the evidence and Bayes Net provided
def get_probability_given_parents(V, v, evidence, bn):
    parents = get_parents(V, bn)
    probability = 0
    if parents is None:
        cpt = bn["CPT("+V+")"]
        probability = cpt[v]
    else:
        cpt = bn["CPT("+V+"|"+parents+")"]
        values = v
        for parent in parents.split(","):
            separator = "|" if values == v else ","
            values = values + separator + evidence[parent]
        probability = cpt[values]

    return probability


# returns the domain values of random variable 'V' given Bayes Net 'bn'
def get_domain_values(V, bn):
    domain_values = []

    for key, cpt in bn.items():
        if key == "CPT("+V+")":
            domain_values = list(cpt.keys())

        elif key.startswith("CPT("+V+"|"):
            for entry, prob in cpt.items():
                value = entry.split("|")[0]
                if value not in domain_values:
                    domain_values.append(value)

    if len(domain_values) == 0:
        print("ERROR: Couldn't find values of variable "+str(V))
        exit(0)

    return domain_values


# returns the number of probabilities (full enumeration) of random variable 'V',
# which is currently used to calculate the penalty of the BIC scoring function.
def get_number_of_probabilities(V, bn):
    for key, cpt in bn.items():
        if key == "CPT("+V+")":
            return len(cpt.keys())

        elif key.startswith("CPT("+V+"|"):
            return len(cpt.items())


# returns the index of random variable 'V' given Bayes Net 'bn'
def get_index_of_variable(V, bn):
    for i in range(0, len(bn["random_variables"])):
        variable = bn["random_variables"][i]
        if V == variable:
            return i

    print("ERROR: Couldn't find index of variable "+str(V))
    exit(0)


# returns a normalised probability distribution of the provided counts,
# where counts is a dictionary of domain_value-counts
def normalise(counts):
    _sum = 0
    for value, count in counts.items():
        _sum += count

    distribution = {}
    for value, count in counts.items():
        p = float(count/_sum)
        distribution[value] = p

    return distribution


# requires the following dependency: pip install networkx
def has_cycles(edges):
    print("\nDETECTING cycles in graph %s" % (edges))
    G = nx.DiGraph(edges)

    cycles = False
    for cycle in nx.simple_cycles(G):
        print("Cycle found:"+str(cycle))
        cycles = True

    if cycles is False:
        print("No cycles found!")
    return cycles