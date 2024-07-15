with open("data.csv", "r") as f, open("data2.csv", "w") as f2:
  for i,line in enumerate(f):
    if i < 1000:
      f2.write(line)
    else:
      break