from FinStat import Corr
from FinStat import factors

fs = Corr("data")
fs.load_data()
pca = factors(fs)

print(pca.fit())
