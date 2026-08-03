# Foresight AI

**An early-warning system for company failure.** It spots a company sliding toward
financial trouble months before the official numbers admit it, and hands you a short,
ranked list of who to worry about instead of a haystack to search.

Live app: [foresightai.streamlit.app](https://foresightai.streamlit.app). Every number on
this page is worked out step by step in the notebooks (`notebooks/`), starting with
[the problem and the data](notebooks/01_data_and_altman.ipynb).

## The problem

If a company you depend on quietly runs out of money, you lose too. A bank loses the loan.
A supplier loses the goods it shipped on credit. An investor loses the position. A vendor
loses a customer that owes it money.

The catch is timing. Most people find out a company is in trouble the same way the public
does: when its official financial results are filed, which happens roughly once a year and
lands months after the trouble actually began. By then the loan is out, the goods are
shipped, the money is gone.

So the real job is not "predict bankruptcy in theory." It is: **should you change your
decision today, because this company's risk is changing right now?** Tighten the credit
line, ask for payment upfront, trim the exposure - while it still matters.

## The tool everyone uses today, and why it falls short

The industry's default early-warning check is the **Altman Z-score**, a formula from 1968.
It takes a handful of numbers from a company's accounts (how much it owes versus what it
owns, whether it is profitable, how much cash it is tying up) and boils them into a single
score. A low score means "looks financially fragile." It is the smoke detector of corporate
credit: simple, everywhere, decades old.

The problem is that this smoke detector is **noisy**. It goes off far too often. To be sure
you catch the genuinely failing companies, you have to investigate a huge pile of healthy
ones alongside them. In our test, catching about half of all the truly distressed companies
with the Altman screen meant flagging **1,586 companies to find just 141 real ones** - your
analysts wade through roughly 1,400 false alarms to get there. That investigation pile, the
"review queue," is the real, recurring, expensive cost of the old way.

## What Foresight AI does differently

Two things.

**1. It ranks far more sharply, so the review queue shrinks.** Instead of a noisy pass/fail
line, Foresight scores and ranks every company by risk. Catching the *same* real distress
cases as the Altman screen above, it flags only **143 companies - and 140 of them are
genuinely in trouble**. Same catch, one-tenth the pile to investigate.

| To catch the same real failures | Companies you must investigate | Real distress among them | Hit rate |
|---|---|---|---|
| Altman screen (today's standard) | 1,586 | 141 | 8.9% |
| Foresight AI | 143 | 140 | 97.9% |

Read the last column as: with the old screen, fewer than 1 in 10 flagged companies is
actually in trouble; with Foresight, almost every company it flags is. Nearly no wasted
investigation.

**2. It reads the outside world, not just the accounts.** Financial statements describe what
already happened, once a year. But distress shows up in the open long before it reaches the
accounts: the CFO resigns, layoffs are announced, the news turns, employees start rating the
company badly, hiring freezes. Foresight tracks four of these live "digital signals"
alongside the financials, on one shared 0-to-100 scale, and shows them side by side. When the
outside world looks worse than the last set of accounts, that gap is the early warning.

## A concrete example

Take Ola Electric. Long before its annual accounts confirmed anything was wrong, the public
signals were already flashing: the CFO resigned, a 5% workforce layoff was announced, and
sales were visibly collapsing in the news. Foresight's Case Study view lets you watch those
signals move month by month, ahead of the filing that eventually made it official. That is
the difference between acting early and reading about it after.

## The results, in plain terms

Everything below is measured, not asserted, on ~10,000 real companies with a known outcome.

- **One-tenth the review queue, roughly 11x sharper.** Same real failures caught, but you
  investigate 143 companies instead of 1,586. Analyst time is the scarce resource in a credit
  team; this is where the money is saved.

- **It ranks the rare thing well: a score of 0.780 out of 1.** Only about 4% of companies in
  the data actually fail, so failures are rare needles in a large haystack. The standard grade
  for "how well did you rank the needles to the top" is called PR-AUC. Pure guessing scores
  0.039 here; Foresight scores **0.780 - about 20 times better than chance.**

- **Why we don't quote "accuracy."** When only 4% fail, a lazy model that declares *everyone*
  healthy is "96% accurate" and catches exactly zero failures. Accuracy rewards doing nothing
  on rare events, so it is the wrong yardstick. We grade on the ranking quality instead.

- **The score means what it says.** When Foresight puts a company at 81, that reflects roughly
  an 81% modelled chance of distress, not just "somewhere near the top." A committee can read
  the number as a real probability and act on it. (In testing, the stated probabilities are
  off by only about 1.5 points on average.)

## Putting a rupee value on the decision

Foresight turns the risk score into a money decision. You tell it two numbers: what one
missed failure costs you (the loss when a borrower you didn't flag collapses) and what one
investigation costs (analyst time to check a company). It then finds the cheapest review
policy for *your* numbers, live.

Worked example, at Rs 50 lakh per missed failure and Rs 1 lakh per investigation:

> Review your riskiest 18% of companies, catch 90% of all the distress out there, for a total
> expected cost of **Rs 26.6 crore**. The best the old Altman screen can manage on the same
> money math is **Rs 65.7 crore**. That is a **Rs 39 crore swing** - and it recalculates
> instantly when you change the numbers, in the Review Economics tab.

## Why the "smarter AI" is the benchmark, not the shipped product

This is the honest, and most interesting, part.

We built a modern machine-learning model as well as the classic Altman engine, and tested
both on real, named bankruptcies. In controlled tests the ML model wins decisively - that is
where the 11x above comes from. But it was trained on European (Polish) companies, and a
model like this learns patterns only within the range of numbers it has seen. Point it at a
live Indian balance sheet, with values outside anything in its training, and it stops
separating the healthy from the doomed:

| Company | What actually happened | ML model's risk rank |
|---|---|---|
| TCS | healthy | 2% |
| Infosys | healthy | 47% |
| Jet Airways | went bankrupt | 59% |
| RCom | in distress | 93% |

Bankrupt Jet Airways lands barely above healthy Infosys - too muddy to act on. The old
Altman formula, being simple arithmetic rather than a learned pattern, has no "training range"
to fall outside of, so it keeps separating the same companies cleanly:

| Company | What actually happened | Altman score | Verdict |
|---|---|---|---|
| TCS | healthy | +10.96 | Safe |
| HUL | healthy | +4.13 | Safe |
| Nestle | healthy | +3.22 | Safe |
| Jet Airways | went bankrupt | -17.25 | Distress |

So Foresight **serves live scores using the dependable Altman engine**, and keeps the ML
model as the benchmark that proves the approach beats the textbook screen 11x over. The
lesson is worth stating plainly: the flashiest model is not automatically the right one to
ship, and we have the measurements to show which one holds up on a real company.

## What it scores today

The live dashboard scores six current Indian companies, riskiest first. "Combined" fuses the
financial score with the four digital signals; scores run 0 to 100, higher = more risk.

| Company | Sector | Financial | Digital | Combined | Band |
|---|---|---|---|---|---|
| SpiceJet | Airline | 99 | 65 | 86 | Critical |
| Ola Electric | Electric Vehicles | 90 | 74 | 83 | Critical |
| Vodafone Idea | Telecom | 87 | 41 | 68 | Elevated Risk |
| Vedanta | Metals & Mining | 48 | 16 | 36 | Watch |
| TCS | IT Services | 1 | 43 | 18 | Healthy |
| Paytm | Fintech | 10 | 13 | 11 | Healthy |

TCS shows why reading both sides matters: its finances are pristine (1), but its digital
signal sits at 43 on workforce-related news - a small early wobble you would miss if you only
ever looked at the accounts.

## What you actually see in the app

Four tabs:

- **Company Analysis** - one risk gauge, the financial ratios behind it, the four live
  signals, and a plain-English "why this score" a credit committee can act on.
- **Portfolio Monitor** - your whole book of companies ranked by risk, worst first.
- **Case Study** - the signals moving month by month ahead of a real failure (the Ola
  Electric story above).
- **Review Economics** - the cheapest review policy for your own cost of a miss and a check.

Plus a one-click PDF report per company and an AI-written analyst summary in readable English.

## Honest scope

This is a decision-support tool meant to sit next to a professional analyst, not replace one.
Risk bands come from a calibrated probability. Some signals in the demo use publicly available
historical data and are labelled as illustrative, because live licensed feeds are what a
production version would connect to. The point of the project is the method and the evidence
behind every choice - the full reasoning is worked through in the three notebooks.
