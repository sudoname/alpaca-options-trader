# 1DTE vs 2DTE Strategy Comparison

## Executive Summary

**1DTE (Current):** High risk/reward, faster trades, more gamma
**2DTE (Alternative):** Lower risk, more time, more stable

---

## Key Differences

| Characteristic | 1DTE | 2DTE |
|----------------|------|------|
| **Time to Expiration** | 0-1 days | 1-2 days |
| **Typical Premium** | $0.50 - $1.50 | $1.00 - $3.00 |
| **Theta Decay** | -50% to -80% per day | -30% to -50% per day |
| **Gamma** | Very high | Moderate |
| **Delta** | Changes rapidly | More stable |
| **Profit Potential** | 20-50% in hours | 15-30% over 1-2 days |
| **Win Rate (Expected)** | 60-70% | 70-80% |
| **Time in Trade** | 4-6 hours | 1-2 days |
| **Overnight Risk** | Rare (close by 3:45 PM) | Often holds overnight |

---

## Detailed Comparison

### Premium Costs

#### 1DTE
**Example:** SPY at $600, targeting 0.35 delta OTM
- **CALL Strike:** $602 (0.3% OTM)
- **Premium:** $0.60 - $1.20
- **Cost:** $60 - $120 per contract
- **Reason:** Very little time value left

#### 2DTE
**Example:** Same setup
- **CALL Strike:** $602 (0.3% OTM)
- **Premium:** $1.50 - $2.50
- **Cost:** $150 - $250 per contract
- **Reason:** Extra day of time value

**Winner for Low Capital:** 1DTE (lower entry cost)

---

### Theta Decay

#### 1DTE
- **Decay Rate:** -0.15 to -0.25 per hour
- **Total Daily Decay:** -50% to -80%
- **Impact:** Must be right quickly
- **Strategy:** Intraday scalping

**Example:**
- Entry at 10:00 AM: $1.00
- By 2:00 PM (4 hours): $0.70 (if flat)
- By Close: $0.50 (if flat)

#### 2DTE
- **Decay Rate:** -0.05 to -0.10 per hour
- **Total Daily Decay:** -30% to -50%
- **Impact:** More time to be right
- **Strategy:** Swing trade (1-2 days)

**Example:**
- Entry Day 1: $2.00
- End of Day 1: $1.40 (if flat)
- End of Day 2: $0.80 (if flat)

**Winner for Beginners:** 2DTE (more forgiving)

---

### Gamma Risk

#### 1DTE
- **Gamma:** Very high (0.10 - 0.20)
- **Impact:** Delta changes rapidly
- **Behavior:**
  - Small SPY move = big option move
  - Can go 0.30 delta → 0.70 delta quickly
  - Whipsaw risk is high

**Example:**
- SPY moves +0.5% in your favor
- Option gains 25-40%
- But SPY reverses -0.5%
- Option drops 30-50%

#### 2DTE
- **Gamma:** Moderate (0.05 - 0.10)
- **Impact:** Delta changes slowly
- **Behavior:**
  - More predictable price action
  - Less whipsaw
  - Smoother P&L curve

**Example:**
- SPY moves +0.5% in your favor
- Option gains 15-25%
- If SPY reverses -0.5%
- Option drops 20-30%

**Winner for Stability:** 2DTE

---

### Win Rate Expectations

#### 1DTE
**Expected Win Rate:** 60-70%
- **Pros:**
  - High profit potential (20-50%)
  - Quick feedback (same day)
- **Cons:**
  - Theta decay works against you
  - Must be right quickly
  - Less room for error

**Our Backtest:** 67.6% win rate

#### 2DTE
**Expected Win Rate:** 70-80%
- **Pros:**
  - More time to be right
  - Can hold through small reversals
  - Lower theta decay pressure
- **Cons:**
  - Lower profit per winner (15-30%)
  - Overnight risk
  - Slower feedback

**Winner for Win Rate:** 2DTE (10% higher expected)

---

### Trading Style

#### 1DTE - Day Trading
```
Timeline:
10:00 AM - Enter trade
10:15 AM - First check
11:00 AM - Check (early stop if -20%)
12:00 PM - Check
1:00 PM  - Check
2:00 PM  - Check
3:00 PM  - Check
3:45 PM  - Close if not already exited

Strategy:
- Scalp 20% quickly
- Cut losers fast (-20% or -30%)
- Close by end of day
- No overnight risk
```

#### 2DTE - Swing Trading
```
Timeline:
Day 1:
10:00 AM - Enter trade
2:00 PM  - Check
4:00 PM  - Hold overnight or take profit

Day 2:
10:00 AM - Check
2:00 PM  - Check
3:45 PM  - Must close (expiration day)

Strategy:
- Target 15-25% over 1-2 days
- Can ride through intraday noise
- May hold overnight
- Overnight gap risk
```

**Winner for Simplicity:** 1DTE (no overnight)

---

### Profit Targets & Stop Losses

#### 1DTE Strategy
- **Profit Target:** 20%
- **Stop Loss:** -30%
- **Early Stop:** -20% before 11 AM
- **Trailing Stop:** 10% from peak after 15%
- **Risk/Reward:** 1:0.67 (30% risk for 20% gain)

#### 2DTE Strategy (Recommended)
- **Profit Target:** 25%
- **Stop Loss:** -40%
- **Early Stop:** -25% on Day 1
- **Trailing Stop:** 12% from peak after 20%
- **Risk/Reward:** 1:0.625 (40% risk for 25% gain)

**Why higher targets for 2DTE?**
- More time = larger potential moves
- Can capture bigger trends
- Higher premium cost needs higher profit

---

### Capital Requirements

#### 1DTE
**Per Trade:**
- **Premium:** $50 - $150
- **Recommended Account:** $2,500+ (2% risk)
- **Contracts:** 1-2 per trade

**$500 Account:**
- Max risk per trade: $50 (10%)
- Can trade 1 contract
- Tight position sizing

#### 2DTE
**Per Trade:**
- **Premium:** $150 - $300
- **Recommended Account:** $5,000+ (2% risk)
- **Contracts:** 1 per trade

**$500 Account:**
- Max risk per trade: $150 (30%)
- Can trade 1 contract
- Very tight sizing
- Higher risk per trade

**Winner for Small Accounts:** 1DTE (lower cost)

---

## Expected Performance Comparison

### 1DTE Strategy (Current)

**Backtest Results (Jan-Oct 2025):**
```
Win Rate: 67.6%
Trades Per Month: ~11 (after filtering)
Average Win: $21
Average Loss: -$6
Profit Factor: 7.00
Monthly Return: +1-2%
```

**Projected Annual:**
- Trades: ~132 per year
- Wins: 89 (67.6%)
- Losses: 43
- Expected Return: 12-24% per year

### 2DTE Strategy (Estimated)

**Expected Performance:**
```
Win Rate: 75% (higher)
Trades Per Month: ~8-10 (less frequent)
Average Win: $35-40
Average Loss: -$15-20
Profit Factor: 5.0-6.0
Monthly Return: +2-3%
```

**Projected Annual:**
- Trades: ~100 per year
- Wins: 75 (75%)
- Losses: 25
- Expected Return: 20-30% per year

**Winner for Returns:** 2DTE (potentially higher)

---

## Pros & Cons Summary

### 1DTE Advantages ✅
1. **Lower capital required** ($50-120 per trade)
2. **No overnight risk** (close same day)
3. **Quick feedback** (know result in hours)
4. **More trading opportunities** (trade every day)
5. **Simpler timing** (no overnight decisions)

### 1DTE Disadvantages ❌
1. **Brutal theta decay** (-50-80% per day)
2. **High gamma risk** (whipsaw)
3. **Must be right quickly** (no time for recovery)
4. **Lower win rate** (60-70% vs 75%)
5. **More stressful** (constant monitoring)

### 2DTE Advantages ✅
1. **Higher win rate** (70-80%)
2. **More time to be right** (1-2 days)
3. **Lower theta decay** (-30-50% per day)
4. **More stable delta** (less gamma)
5. **Larger profit potential** (25-40%)

### 2DTE Disadvantages ❌
1. **Higher capital required** ($150-300 per trade)
2. **Overnight risk** (gap up/down)
3. **Slower feedback** (1-2 days)
4. **Fewer trades** (need more time per trade)
5. **Weekend risk** (if entering Friday)

---

## Recommended Modifications for 2DTE

If you want to switch to 2DTE, here are the changes needed:

### 1. Change DTE Filter
```python
# In spy_1dte_strategy.py, line 204
# Old:
if days_to_exp <= 1:  # 0DTE or 1DTE

# New:
if days_to_exp <= 2 and days_to_exp >= 1:  # 1DTE or 2DTE
```

### 2. Adjust Profit Target
```python
# Line 69
self.profit_target = 0.25  # 25% instead of 20%
```

### 3. Adjust Stop Loss
```python
# Line 70
self.stop_loss = -0.40  # -40% instead of -30%
```

### 4. Adjust Early Stop
```python
# Line 72
self.early_stop_loss = -0.25  # -25% instead of -20%
```

### 5. Change Monitoring
```python
# Line 74
self.monitor_interval = 1800  # Check every 30 minutes (instead of 15)
```

### 6. Update Close Time
```python
# In monitor_and_close(), around line 510
# Old:
if now.hour >= 15 and now.minute >= 45:  # Close at 3:45 PM

# New:
# For Day 1: Hold overnight
# For Day 2: Close at 3:45 PM
days_held = (now - entry_time).days
if days_held >= 1 and now.hour >= 15 and now.minute >= 45:
```

---

## Which Should You Choose?

### Choose 1DTE if:
- ✅ You have a small account ($500 - $2,500)
- ✅ You don't want overnight risk
- ✅ You can monitor during the day
- ✅ You prefer quick feedback
- ✅ You want more trading opportunities
- ✅ You're comfortable with high gamma

### Choose 2DTE if:
- ✅ You have a larger account ($2,500+)
- ✅ You're OK with overnight risk
- ✅ You want higher win rate (75%+)
- ✅ You prefer swing trading
- ✅ You want larger profit targets (25-40%)
- ✅ You can't monitor intraday constantly

---

## Hybrid Approach

**Best of Both Worlds:**

```python
Strategy Rules:
1. Enter 2DTE options at 10:00 AM
2. If up 25%+ same day → Close (scalp like 1DTE)
3. If flat/small gain → Hold overnight
4. On Day 2: Close at 3:45 PM or 25% profit
5. Stop loss: -25% Day 1, -40% Day 2
```

**Benefits:**
- Lower theta decay than 1DTE
- Can still scalp if opportunity arises
- More time for trade to work
- Higher win rate potential

**Drawbacks:**
- Higher premium cost
- Overnight risk
- More complex decision-making

---

## Performance Projections

### Small Account ($500)

**1DTE:**
- Trades/month: 11
- Cost/trade: $80
- Win rate: 67.6%
- Expected monthly: +$10-20 (2-4%)
- **Annual: 24-48%**

**2DTE:**
- Trades/month: 8
- Cost/trade: $200
- Win rate: 75%
- Expected monthly: +$20-30 (4-6%)
- **Annual: 48-72%**
- ⚠️ **But higher risk** (40% of account per trade)

### Medium Account ($2,500)

**1DTE:**
- Contracts: 2-3
- Cost/trade: $200-300
- Win rate: 67.6%
- Expected monthly: +$50-75 (2-3%)
- **Annual: 24-36%**

**2DTE:**
- Contracts: 1-2
- Cost/trade: $300-400
- Win rate: 75%
- Expected monthly: +$75-100 (3-4%)
- **Annual: 36-48%**

**Winner for Medium Account:** 2DTE

### Large Account ($10,000+)

**Both work well!**
- Can run 1DTE for daily income
- Add 2DTE for swing trades
- Diversify expiration dates
- Reduce single-trade risk

---

## Final Recommendation

### For Your $500 Account:

**STICK WITH 1DTE** because:
1. Lower capital per trade ($60-120)
2. No overnight risk
3. 67.6% win rate is strong
4. Proven backtest results
5. You can trade more frequently

**Consider 2DTE when:**
- Account grows to $2,500+
- You want to reduce time commitment
- You're comfortable with overnight risk
- You want higher win rate

---

## Summary Table

| Factor | 1DTE | 2DTE | Winner |
|--------|------|------|--------|
| **Win Rate** | 67.6% | ~75% | 2DTE |
| **Capital Required** | $60-120 | $150-300 | 1DTE |
| **Time Commitment** | High (intraday) | Low (swing) | 2DTE |
| **Theta Decay** | -50-80% | -30-50% | 2DTE |
| **Overnight Risk** | None | Yes | 1DTE |
| **Profit Target** | 20% | 25% | 2DTE |
| **Trading Frequency** | Daily | 2-3x/week | 1DTE |
| **Best for Beginners** | Harder | Easier | 2DTE |
| **Best for Small Account** | Yes | No | 1DTE |

---

## Code to Create 2DTE Version

Want me to create a 2DTE version? I can:
1. Copy `spy_1dte_strategy.py` → `spy_2dte_strategy.py`
2. Adjust all parameters
3. Add overnight logic
4. Create new backtest
5. Compare results side-by-side

**Just ask and I'll implement it!** 🚀

---

*Last Updated: October 14, 2025*
*Current Strategy: 1DTE (Recommended for $500 account)*
