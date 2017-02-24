from primo2 import io
from primo2.inference import dynamic

dbn = io.DBNSpec.parse('dbn-example.conf')
pfb = dynamic.PriorFeedbackExact(dbn)

for x in range(15):
	if x % 3 == 0:
		pfb.unroll({'fb_function': 'understanding'}, soft_posteriors=True)
		print(x, '!', pfb.marginal_probabilities('understanding'))
	else:
		pfb.unroll()
		print(x, pfb.marginal_probabilities('understanding'))
