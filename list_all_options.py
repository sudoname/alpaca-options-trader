from spy_qqq_hybrid_strategy import SPYQQQHybridStrategy
from schwab.client import Client
from datetime import datetime

strategy = SPYQQQHybridStrategy()
target_date = datetime(2025, 10, 24).date()

print('=' * 80)
print('ALL AVAILABLE OPTIONS - SPY & QQQ')
print('=' * 80)
print(f'Expiration: Friday, October 24, 2025 (7 days)')
print(f'Filters: Delta 0.25-0.35 | Max Premium $6.00 | Min Volume 100 | Min OI 500')
print('=' * 80)

for ticker in ['SPY', 'QQQ']:
    print(f'\n{ticker} OPTIONS:')
    print('-' * 80)

    # Get current price
    response = strategy.client.get_quote(ticker)
    data = response.json()[ticker]['quote']
    current_price = data['lastPrice']
    print(f'{ticker} Price: ${current_price:.2f}\n')

    # Get PUT option chain
    response = strategy.client.get_option_chain(
        ticker,
        contract_type=Client.Options.ContractType.PUT,
        strike_count=60,
        include_underlying_quote=True,
        from_date=target_date,
        to_date=target_date
    )

    chain_data = response.json()

    # Collect PUT options
    put_options = []
    if 'putExpDateMap' in chain_data:
        for exp_date, strikes in chain_data['putExpDateMap'].items():
            for strike_price, contracts in strikes.items():
                for contract in contracts:
                    strike = contract.get('strikePrice', 0)
                    delta = abs(contract.get('delta', 0))
                    ask = contract.get('ask', 0)
                    bid = contract.get('bid', 0)
                    volume = contract.get('totalVolume', 0)
                    oi = contract.get('openInterest', 0)

                    is_otm = strike < current_price

                    if (is_otm and ask > 0 and ask <= 6.0 and
                        0.25 <= delta <= 0.35 and
                        volume >= 100 and oi >= 500):

                        delta_score = (100 - abs((delta - 0.30) * 200)) * 2
                        volume_score = min(volume / 1000 * 20, 60)
                        oi_score = min(oi / 1000 * 15, 40)
                        spread = ask - bid
                        spread_pct = spread / ask if ask > 0 else 1
                        spread_score = max(40 - (spread_pct * 100), 0)
                        risk_reward_score = max(30 - (ask * 3), 0)
                        total_score = delta_score + volume_score + oi_score + spread_score + risk_reward_score

                        put_options.append({
                            'strike': strike,
                            'delta': delta,
                            'ask': ask,
                            'bid': bid,
                            'volume': volume,
                            'oi': oi,
                            'spread_pct': spread_pct * 100,
                            'score': total_score,
                            'symbol': contract.get('symbol')
                        })

    put_options.sort(key=lambda x: x['score'], reverse=True)

    if put_options:
        print(f'Found {len(put_options)} PUT options:\n')
        print('Rank  Strike    Delta    Premium  Volume    OI        Spread%  Score')
        print('-' * 85)

        for i, opt in enumerate(put_options, 1):
            print(f'{i:<5} ${opt["strike"]:<7.2f} {opt["delta"]:.4f}   ${opt["ask"]:<7.2f} {opt["volume"]:<9.0f} {opt["oi"]:<9.0f} {opt["spread_pct"]:<7.1f}% {opt["score"]:<7.1f}')

        best = put_options[0]
        print(f'\n>>> BEST PUT (Highest Profit Score):')
        print(f'    Strike: ${best["strike"]:.2f} | Premium: ${best["ask"]:.2f} | Delta: {best["delta"]:.4f}')
        print(f'    Cost: ${best["ask"] * 100:.2f} | Volume: {best["volume"]:.0f} | OI: {best["oi"]:.0f}')
        print(f'    Profit Score: {best["score"]:.1f}')
    else:
        print('No PUT options found.')

    # Get CALL option chain
    response = strategy.client.get_option_chain(
        ticker,
        contract_type=Client.Options.ContractType.CALL,
        strike_count=60,
        include_underlying_quote=True,
        from_date=target_date,
        to_date=target_date
    )

    chain_data = response.json()

    call_options = []
    if 'callExpDateMap' in chain_data:
        for exp_date, strikes in chain_data['callExpDateMap'].items():
            for strike_price, contracts in strikes.items():
                for contract in contracts:
                    strike = contract.get('strikePrice', 0)
                    delta = abs(contract.get('delta', 0))
                    ask = contract.get('ask', 0)
                    bid = contract.get('bid', 0)
                    volume = contract.get('totalVolume', 0)
                    oi = contract.get('openInterest', 0)

                    is_otm = strike > current_price

                    if (is_otm and ask > 0 and ask <= 6.0 and
                        0.25 <= delta <= 0.35 and
                        volume >= 100 and oi >= 500):

                        delta_score = (100 - abs((delta - 0.30) * 200)) * 2
                        volume_score = min(volume / 1000 * 20, 60)
                        oi_score = min(oi / 1000 * 15, 40)
                        spread = ask - bid
                        spread_pct = spread / ask if ask > 0 else 1
                        spread_score = max(40 - (spread_pct * 100), 0)
                        risk_reward_score = max(30 - (ask * 3), 0)
                        total_score = delta_score + volume_score + oi_score + spread_score + risk_reward_score

                        call_options.append({
                            'strike': strike,
                            'delta': delta,
                            'ask': ask,
                            'bid': bid,
                            'volume': volume,
                            'oi': oi,
                            'spread_pct': spread_pct * 100,
                            'score': total_score,
                            'symbol': contract.get('symbol')
                        })

    call_options.sort(key=lambda x: x['score'], reverse=True)

    if call_options:
        print(f'\n{ticker} CALL OPTIONS:')
        print('-' * 80)
        print(f'Found {len(call_options)} CALL options:\n')
        print('Rank  Strike    Delta    Premium  Volume    OI        Spread%  Score')
        print('-' * 85)

        for i, opt in enumerate(call_options, 1):
            print(f'{i:<5} ${opt["strike"]:<7.2f} {opt["delta"]:.4f}   ${opt["ask"]:<7.2f} {opt["volume"]:<9.0f} {opt["oi"]:<9.0f} {opt["spread_pct"]:<7.1f}% {opt["score"]:<7.1f}')

        best = call_options[0]
        print(f'\n>>> BEST CALL (Highest Profit Score):')
        print(f'    Strike: ${best["strike"]:.2f} | Premium: ${best["ask"]:.2f} | Delta: {best["delta"]:.4f}')
        print(f'    Cost: ${best["ask"] * 100:.2f} | Volume: {best["volume"]:.0f} | OI: {best["oi"]:.0f}')
        print(f'    Profit Score: {best["score"]:.1f}')

print('\n' + '=' * 80)
