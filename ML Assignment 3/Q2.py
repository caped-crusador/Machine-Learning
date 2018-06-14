from lea import *
flip1 = Lea.fromValFreqs(("head", 67),("tail", 33))
flip1

P(flip1 == "head")
Pf(flip1 == "head")

# 10 random flips
flip1.random(10)


# --------------------------------------------------
Alternator_broken = Lea.boolProb(1, 1000)
FanBelt_broken = Lea.boolProb(2, 100)
Gas = Lea.boolProb(100, 100)
starter = Lea.boolProb(100, 100)

Battery_Charging = Lea.buildCPT(
                    (Alternator_broken & FanBelt_broken, Lea.boolProb(0, 1000)),
                    (Alternator_broken & ~FanBelt_broken, Lea.boolProb(0, 1000)),
                    (~Alternator_broken & FanBelt_broken, Lea.boolProb(0, 1000)),
                    (~Alternator_broken & ~FanBelt_broken, Lea.boolProb(995, 1000)))

Battery_flat = Lea.buildCPT(
                (Battery_Charging, Lea.boolProb(10, 100)),
                (~Battery_Charging, Lea.boolProb(90, 100)))

Car_starts = Lea.buildCPT((Gas & starter & ~Battery_flat, Lea.boolProb(95, 100)),
                          (Gas & starter & Battery_flat, Lea.boolProb(0, 100)),
                          (Gas & ~starter & ~Battery_flat, Lea.boolProb(0, 100)),
                          (Gas & ~starter & Battery_flat, Lea.boolProb(0, 100)),
                          (~Gas & starter & ~Battery_flat, Lea.boolProb(0, 100)),
                          (~Gas & starter & Battery_flat, Lea.boolProb(0, 100)),
                          (~Gas & ~starter & ~Battery_flat, Lea.boolProb(0, 100)),
                          (~Gas & ~starter & Battery_flat, Lea.boolProb(0, 100)),)

print("Probability that the Alternator is Broken, given the Car won't start : ",
      Pf(Alternator_broken.given(~Car_starts)), "\n")
print("Probability that the Fan Belt is Broken, given the Car won't start : ",
      Pf(FanBelt_broken.given(~Car_starts)), "\n")
print("Probability that the Fan Belt is Broken, given the Car won't start and Alternator is broken : ",
      Pf(FanBelt_broken.given(~Car_starts & Alternator_broken)), "\n")
print("Probability that the Alternator and the Fan Belt are broken, given the Car won't start : ",
      Pf((Alternator_broken & FanBelt_broken).given(~Car_starts)), "\n")
