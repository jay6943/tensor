import classicification_def

model = classicification_def.LogisticRegression()

for parameter in model.parameters():
  print(parameter)
