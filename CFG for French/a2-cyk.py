#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, json, string, itertools, sys


# In[2]:


NAME_STACK = []
GRAMMAR_DICT = {}
FNAL_CNF_GRAMMAR = []
CNF_NODES = []


# In[3]:


# read in grammar file
def read_grammar():
    
    rules = []
    with open(CFG_FILE) as g_file:
        lines = g_file.readlines()
        
        for line in lines:
            if(len(line) > 1):
                rules.append(str(line).lower())
    
    return rules


# In[4]:


# function to split the grammar rules that have | to multiple individual rules
def split_grammar(rules):
    new_rules = []
    
    done = True
    
    for rule in rules:
        left, right = rule.split(" -> ")
        
        if "|" in right:
            done = False
            right = right.split("|")
            
            for right_item in right:
                new_rules.append(left + " -> " + right_item)
                
        else:
            new_rules.append(rule)
            
    return new_rules, done


# In[5]:


# function to do the first CNF processing step:
# A -> B C D ::
# A -> X1 D
# X1 -> B C
def to_cnf_1(rules):
    
    done = True
    new_rules = []
        
    for rule in rules:
        
        left, right = rule.split(" -> ")
        
        if len(right.split()) > 2:
            done = False
            
            right = right.split()
            
            new_name = NAME_STACK.pop()
            new_rules.append(left + " -> " + new_name + " " + " ".join(right[2:]))
            new_rules.append(new_name + " -> " + " ".join(right[:2]))
            
        else:
            new_rules.append(rule)
    
    return new_rules, done


# In[6]:


# function to do the second CNF processing step:
# A -> 's' B ::
# A -> X2 B
# X2 -> 's'
def to_cnf_2(rules):
    
    done = True
    
    new_rules = []
        
    for rule in rules:

        left, right = rule.split(" -> ")
        
        right = right.split()
        
        if len(right) >= 2:
            
            if "\"" in right[0] or "\"" in right[0] or "\'" in right[0] or "\'" in right[0]:
                done = False
                new_name = NAME_STACK.pop()
                new_rules.append(left + " -> " + new_name + " " + right[1])
                new_rules.append(new_name + " -> " + " " + right[0])

            elif "\"" in right[1] or "\"" in right[1] or "\'" in right[1] or "\'" in right[1]:
                done = False
                new_name = NAME_STACK.pop()
                new_rules.append(left + " -> " + new_name + " " + right[0])
                new_rules.append(new_name + " -> " + " " + right[1])

            else:
                new_rules.append(rule)
            
        else:
            new_rules.append(rule)
    
    return new_rules, done


# In[7]:


# function to do the second CNF processing step:
# A -> B ::
# for every rule in the grammar that B is in the LHS, copy those rules but replace B with A
def to_cnf_3(rules):
    
    done = True
    
    new_rules = []
    
    for rule in rules:

        left, right = rule.split(" -> ")
        
        split_right = right.split()
        
        if len(split_right) == 1:
            
            if not("\"" in split_right[0] or "\"" in split_right[0] or "\'" in split_right[0] or "\'" in split_right[0] or "\"" in split_right[0] or "\"" in split_right[0] or "\'" in split_right[0] or "\'" in split_right[0]):
                add = ""
                done = False
                
                for i in rules:
                    new_left, new_right = i.split(" -> ")

                    if new_left.strip() == right.strip():
                        if not add == "":
                            add = add + " | " + new_right
                        else:
                            add = new_right
                
                new_rules.append(left + " -> " + add)
                
            else:
                new_rules.append(rule)
        else:
            new_rules.append(rule)
                
        
    return new_rules, done
                


# In[8]:


# extra helper functions
def set_name_stack():
    
    global NAME_STACK
    
    NAME_SATCK = []
    for i in range(20):
        NAME_STACK.append("XXX-"+str(i))
        NAME_STACK.append("ZZZ-"+str(i))
        NAME_STACK.append("YYY-"+str(i))
        NAME_STACK.append("OOO-"+str(i))
        NAME_STACK.append("QQQ-"+str(i))
        NAME_STACK.append("MMM-"+str(i))
        
def add_rule(rules):
    
    rule_arr  = []
    
    for rule in rules:
        rule = rule.replace("->", "").split()
        rule_arr.append(rule)
        
    return rule_arr

def replace(string, char, num, head):
    new_string = ""
    for m in range(num):
        new_string += char
        
    if head:
        return new_string+string
    else:
        return string+new_string
    
def get_key(val):
    rtn = []
    
    for key, value in GRAMMAR_DICT.items():
        value = list(itertools.chain.from_iterable(value))
        if val in value:
            rtn.append(key)
    return rtn

class Node:
    def __init__(self, symbol, left, right=None):
        self.symbol = symbol
        self.left = left
        self.right = right


# In[9]:


# function to turn the CFG to CNF form
def process_grammar():

    set_name_stack()

    rules = split_grammar(read_grammar())[0]
    final_rules = []
    
    done_total = False
    
    while not done_total:
        done_1 = False
        while not done_1:
            rules, done_1 = to_cnf_1(rules)

        done_2 = False
        while not done_2:
            done_total = False
            rules, done_2 = to_cnf_2(rules)

        done_3 = False
        while not done_3:
            rules, done_3 = to_cnf_3(rules)

        rules, done_total = split_grammar(rules)
        
    return rules


# In[10]:


def create_grammar():
    
    global GRAMMAR_DICT
    global CNF_NODES

    GRAMMAR_DICT = {}
    CNF_NODES = []
    
    rules = split_grammar(read_grammar())[0]
    
    new_rules = []
    
    for rule in rules:
        rule = rule.replace("->", "").split()
        new_rules.append(rule)
    
    for rule in new_rules:
        if rule[0] not in GRAMMAR_DICT:
            GRAMMAR_DICT[rule[0]] = []
        GRAMMAR_DICT[rule[0]].append(rule[1:])


# In[11]:


# function to find a path between two given nodes in the original CFG
new_nodes = []
def find_parents(nodes, end):
    global new_nodes
    done = True
    new_nodes = []
    
    for i in nodes:
        last = i[0]
        
        priv = get_key(last)
        
            
        if len(priv) > 0 and not last == end:
            done = False
            for j in priv:
                new_list = i.copy()
                new_list.insert(0,j)
                new_nodes.append(new_list)
        else:
#             done = False
            new_nodes.append(i)

    if not done:
        find_parents(new_nodes, end)
        
    
    return None

def find_path_between_nodes(start, end):
    rtn = []
    find_parents([[start]], end)

    for i in new_nodes:
        if i[0] == end:
            rtn.append(i)
        
    return rtn


def set_params():
    global CNF_NODES
    global FNAL_CNF_GRAMMAR
    CNF_NODES = []
    FNAL_CNF_GRAMMAR = []
    
def process_sentance():
    sentance = re.sub(r'[^\w\s]','',SENTANCE)
    sentance.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('á', 'a').replace('à', 'a').replace('â', 'a').replace('ó', 'o').replace('ò', 'o').replace('ô', 'o')
    sentance = re.sub(r'´','',sentance)
    sentance = sentance.lower().split()
    return sentance


# In[12]:


def generate_tree(node):
    global CNF_NODES
    new_node = {
        'symbol' : node.symbol,
        'left' : node.left,
        'right' : node.right
    }
        
    CNF_NODES.append(new_node)

    if node.right is None:
        return " { " + str(node.symbol) + " " +  str(node.left) +" } "
    return " [ "+ str(node.symbol) + " " + generate_tree(node.left) + " " + generate_tree(node.right)+ " ] "


# In[13]:


def parse(string, FNAL_CNF_GRAMMAR):

# create the table
    parse_table = []
    for i in range(len(string)):
        inner = []
        for j in range(len(string) - i):
            inner.append([])
        parse_table.append(inner)


    for i in range(len(string)):
        for rule in FNAL_CNF_GRAMMAR:
            if "'" + string[i] + "'" == rule[1]:
                parse_table[0][i].append(Node(rule[0], string[i]))


    for words_to_consider in range(2, len(string) + 1):
        for starting_cell in range(0, len(string) - words_to_consider + 1):
            for left_size in range(1, words_to_consider):
                right_size = words_to_consider - left_size

                left_cell = parse_table[left_size - 1][starting_cell]
                right_cell = parse_table[right_size - 1][starting_cell + left_size]

                for rule in FNAL_CNF_GRAMMAR:
                    left_nodes = []
                    for i in left_cell:
                        if i.symbol == rule[1]:
                            left_nodes.append(i)

                    if left_nodes:
                        right_nodes = []
                        for i in right_cell:
                            if i.symbol == rule[2]:
                                right_nodes.append(i)

                        for right in right_nodes:
                            for left in left_nodes:
                                node = Node(rule[0], left, right)
                                parse_table[words_to_consider - 1][starting_cell].append(node)

    return parse_table


# In[14]:


def CFG_tree(nodes):
    
    j = 0
    for i in nodes:
        parent = i[0]
        right = i[1]
        left = i[2]
        
        new_branch = ""
        new_branch = replace(new_branch, "\t".expandtabs(5), j, True)
        
        if left == None:
            new_branch += (str(parent) + " --> [ " + (" -> ".join((find_path_between_nodes("'" + str(right) + "'", str(parent)))[0][1:])) + " ]")
        else:
            First = True
            for m in i[1:]:
                left = m
                if First:
                    First = False
                    new_branch += (str(parent) + " --> " + (" -> ".join((find_path_between_nodes(str(left), str(parent)))[0][1:])))
                else:
                    new_branch += ("     " + (" -> ".join((find_path_between_nodes(str(left), str(parent)))[0][1:])))
                
                new_branch += "\n"
                new_branch = replace(new_branch, "\t".expandtabs(5), j, False)
                new_branch = replace(new_branch, " ", len(str(parent)), False)
                
            
        print(new_branch)
        print("\n".expandtabs(1))

        
        j = j+1
        


# In[15]:


def inner_collapse(rtn):
    done = True
    new_rtn = []
    for index in range(len(rtn)):
        row = rtn[index]
        j = row[1]
        i = str(row[1])

        if ('XXX' in i) or ('YYY' in i) or ('OOO' in i) or ('ZZZ' in i) or ('QQQ' in i) or ('MMM' in i):
            done = False
            row.remove(j)
            new_row = row
            new_row.insert(1, rtn[index+1][1])
            new_row.insert(2, rtn[index+1][2])
            new_rtn.append(new_row)
            new_rtn.extend(rtn[(index+2):])

            return new_rtn, done
        else:
            new_rtn.append(row)

    return new_rtn, done


# In[16]:


def collapse():
    
    rtn = []
    for i in CNF_NODES:
        in_rtn = []
        for j in i.values():
            if j == None or type(j) == str:
                in_rtn.append(j)  
            else:
                in_rtn.append(j.symbol)

        rtn.append(in_rtn)

    new, done = inner_collapse(rtn)
    
    while not done:
        new, done = inner_collapse(new)
        
    return new


# In[17]:


def print_tree(parse_table):
    
    global FNAL_CNF_GRAMMAR
    
    final_nodes = []
    for i in parse_table[-1][0]:
        if i.symbol ==  FNAL_CNF_GRAMMAR[0][0]:
            final_nodes.append(i)

    if final_nodes:
        print("\nThis sentance can be produced using the provided CFG.")

        print("Possible trees (using):")

        cnf_trees = []
        for node in final_nodes:
            print("\nCNF based Tree:\n")
            print(generate_tree(node))

            print("\n\n-------------------------------")
            print("\nCFG based Tree:\n")
            
            nodes = collapse()
            print(CFG_tree(nodes))

    else:
        print("\nThis sentance can not be produced using provided CFG.")


# In[18]:


CFG_FILE = "./french-grammar.txt"
SENTANCE = "Tu regardes la television"


# In[19]:


def main():
    global CFG_FILE
    global SENTANCE

    if (len(sys.argv) == 5) and (sys.argv[1] == '-s') and (sys.argv[3] == '-f'):
        SENTANCE = sys.argv[2]
        CFG_FILE = sys.argv[4]

    if (len(sys.argv) == 3) and (sys.argv[1] == '-s'):
        SENTANCE = sys.argv[2]

    print("\n")
    print("Grammar File: " + CFG_FILE)
    print("Sentance: " + SENTANCE)
    print("\n")

    global CNF_NODES
    global FNAL_CNF_GRAMMAR
    CNF_NODES = []
    FNAL_CNF_GRAMMAR = []
    
    create_grammar()
    cnf_rules = process_grammar()
    FNAL_CNF_GRAMMAR = add_rule(cnf_rules)
    parse_table = parse(process_sentance(), FNAL_CNF_GRAMMAR)
    print_tree(parse_table)


# In[20]:


if __name__ == "__main__":
    main()





