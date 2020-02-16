## 1. Probability basics ##

# Print the first two rows of the data.
print(flags[:2])

most_bars_country = flags.loc[flags['bars'].idxmax()]['name']
highest_population_country = flags.loc[flags['population'].idxmax()]['name']

## 2. Calculating probability ##

total_countries = flags.shape[0]

orange_probability = flags['orange'].sum()/len(flags['orange'])
flags_more_1 = (flags['stripes']>1).sum()
stripe_probability = flags_more_1/total_countries

## 3. Conjunctive probabilities ##

five_heads = .5 ** 5
ten_heads = .5 ** 10
hundred_heads = .5 ** 100

## 4. Dependent probabilities ##

# Remember that whether a flag has red in it or not is in the `red` column.

red_count = flags['red'].sum()
total_count = flags.shape[0]
three_red = red_count/total_count
for i in range(1,3):
    three_red = (red_count - i) / (total_count - i) * three_red

## 5. Disjunctive probability ##

start = 1
end = 18000

hundred_prob = (18000/100)/18000

seventy_prob = round((18000/70),0)/end

## 6. Disjunctive dependent probabilities ##

stripes_or_bars = None
red_or_orange = None
total = flags.shape[0]
red = flags['red'].sum()
orange = flags['orange'].sum()
red_and_orange = flags[(flags['orange']==1) & (flags['red']==1)].shape[0]
red_or_orange = red/total + orange/total - red_and_orange/total

stripes = (flags['stripes']>0).sum()
bars = (flags['bars']>0).sum()
stripes_and_bars = flags[(flags['stripes']>0) & (flags['bars']>0)].shape[0]
stripes_or_bars = stripes/total + bars/total - stripes_and_bars/total

## 7. Disjunctive probabilities with multiple conditions ##

heads_or = 1 - (0.5 ** 3)