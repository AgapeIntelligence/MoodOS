# export_scripted.py
from moodos_v01 import MoodProcessor

bands = [(0.5 + i*1.5, 0.5 + (i+1)*1.5) for i in range(64)]
proc = MoodProcessor(freq_bands_hz=bands)
scripted = torch.jit.script(proc)
scripted.save("moodos_v01_scripted.pt")
print("Saved moodos_v01_scripted.pt – ready for iOS/Android")
