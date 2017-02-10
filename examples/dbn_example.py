from primo2 import dbn
from primo2.inference import exact

d = dbn.create_DBN_from_spec('dbals.conf')
#d.set_unroll_method('PRIOR_FEEDBACK')

# ft = exact.FactorTree.create_jointree(d.B0)
# ft.set_evidence({})
# print('perception', ft.marginals(['perception']).get_potential())
# print('understanding', ft.marginals(['understanding']).get_potential())
# print('acceptance', ft.marginals(['acceptance']).get_potential())
# print('grounding', ft.marginals(['grounding']).get_potential())



## Calculate with primo-legacy# from primo.inference.factor import FactorTreeFactory
# from primo.io import XMLBIF
# bn = XMLBIF.read("/Users/hendrik/Projects/repo/attentivespeaker/resource/data/als/tbals_b0_eval_study_2015-new.xbif")
# factorTreeFactory = FactorTreeFactory()
# factorTree = factorTreeFactory.create_greedy_factortree(bn)
# print(factorTree.calculate_marginal([bn.get_node('perception')]))
# print(factorTree.calculate_marginal([bn.get_node('understanding')]))
# print(factorTree.calculate_marginal([bn.get_node('acceptance')]))
# print(factorTree.calculate_marginal([bn.get_node('grounding')]))



d.unroll()
print
d.unroll({'fb_function': 'understanding'})
print
d.unroll()
print
d.unroll()
print()
d.unroll()
print()
d.unroll()
print()
d.unroll()
print()
print()
d.unroll({'fb_function': 'understanding'})
print()
print()
d.unroll({'fb_function': 'understanding'})
print()
d.unroll({'fb_function': 'understanding'})
print()
d.unroll({'fb_function': 'understanding'})
print()
d.unroll({'fb_function': 'understanding'})
print()
d.unroll()
print()




# import numpy as np
# import primo2.io
# from primo2.inference.exact import FactorTree
# bn = primo2.io.XMLBIFParser.parse("primo2/tests/slippery.xbif")
# bn.get_node('winter').set_cpd(np.array([0.1, 0.9]))
# ft = FactorTree.create_jointree(bn)

# print('winter', ft.marginals(['winter']).get_potential())
# print('rain', ft.marginals(['rain']).get_potential())
# print('road', ft.marginals(['slippery_road']).get_potential())

# print('\nsetting evidence\n')
# ft.set_evidence({'rain': 'false'})
# #ft.set_evidence({'winter': np.array([0.1, 0.9]), 'rain': 'false'}, softPosteriors=True)
# print('winter', ft.marginals(['winter']).get_potential())
# print('rain', ft.marginals(['rain']).get_potential())
# print('road', ft.marginals(['slippery_road']).get_potential())




# import primo2.io
# from primo2.inference.exact import FactorTree

# bn = primo2.io.XMLBIFParser.parse("slippery.xbif")


# ft = FactorTree.create_jointree(bn)
# ft.set_evidence({'winter': 'true'})
# #print(ft.marginals(['rain', 'winter']).get_potential({'rain': 'true'}))
# #print(ft.marginals(['rain']).get_potential())
# print(ft.marginals(['winter']).get_potential())


# import numpy as np
# from primo2.network import BayesianNetwork
# from primo2.nodes import DiscreteNode

# from primo2.inference.exact import FactorTree

# bn = BayesianNetwork()
# cloth = DiscreteNode("cloth", ["green","blue", "red"])
# sold = DiscreteNode("sold")

# bn.add_node(cloth)
# bn.add_node(sold)

# bn.add_edge("cloth", "sold")

# # Bspl. soft evidence aus darwiche (2009, p. 41)

# cloth.set_cpd(np.array([0.3,0.3,0.4]))
# cloth.set_cpd(np.array([0.7,0.25,0.05])) # Was waere, wenn wir soft evidence als prior betrachten
# sold.set_cpd(np.array([[0.4, 0.4, 0.8],
#                         [0.6, 0.6, 0.2]]))

# tree = FactorTree.create_jointree(bn)

# print tree.marginals(["sold"]).get_potential()

# tree.set_evidence({"cloth": np.array([0.7,0.25,0.05])})

# print tree.marginals(["sold"]).get_potential()
