# 必要なライブラリをインポート
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta

# --- ページ設定 ---
st.set_page_config(page_title="VPA分析チャート", layout="wide")

# --- JPX銘柄リストの読み込み ---
@st.cache_data
def load_jpx_list():
    try:
        df_jpx = pd.read_excel("jpx_list.xls")
        df_jpx['コード'] = df_jpx['コード'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_jpx = df_jpx[['コード', '銘柄名', '33業種区分']].copy()
        df_jpx.dropna(subset=['コード', '銘柄名', '33業種区分'], inplace=True)
        df_jpx['display_name'] = df_jpx['銘柄名'] + " (" + df_jpx['コード'] + ")"
        return df_jpx
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"銘柄リストの読み込み中にエラーが発生しました: {e}")
        st.info("`pip install openpyxl xlrd` を実行すると解決する場合があります。")
        return None

jpx_df = load_jpx_list()

# --- サイドバー (ユーザーからの入力) ---
st.sidebar.header("チャート設定")

if jpx_df is not None:
    # ▼▼▼【ここがエラー修正箇所】▼▼▼
    # numpyのint64をPythonのintに変換する
    default_index = int(jpx_df[jpx_df['コード'] == '7974'].index[0])
    company_name = st.sidebar.selectbox(
        '会社名で検索',
        options=jpx_df['display_name'],
        index=default_index # 修正したindexを使用
    )
    # ▲▲▲【エラー修正はここまで】▲▲▲
    user_input = company_name.split('(')[-1].replace(')', '')
else:
    user_input = st.sidebar.text_input("銘柄コード", value="7974").upper().strip()

period_options = { "1ヶ月": "1mo", "6ヶ月": "6mo", "1年": "1y", "5年": "5y", "10年": "10y" }
selected_period = st.sidebar.radio("表示期間を選択", options=list(period_options.keys()), index=2, horizontal=True)

with st.sidebar.expander("バックテスト設定", expanded=True):
    take_profit_percent = st.slider("利益確定率（%）", 0, 100, 20, 1)
    stop_loss_percent = st.slider("ロスカット率（%）", 0, 50, 10, 1)
    commission_percent = st.slider("想定手数料（%）", 0.0, 1.0, 0.1, 0.01)

with st.sidebar.expander("VPA感度設定", expanded=True):
    ma_period = st.slider("MA期間", 10, 100, 25, 5)
    volume_multiplier = st.slider("出来高倍率", 1.5, 3.0, 1.8, 0.1)
    adx_threshold = st.slider("ADX閾値", 10, 40, 20, 1)
    wick_ratio = st.slider("ヒゲ倍率", 1.2, 3.0, 1.5, 0.1)

st.sidebar.header("表示オプション")
show_low_prob = st.sidebar.checkbox('低確率シグナルを表示する', value=False)


# (以降のバックテスト関数、分析ロジック、UI表示は変更なし)
def calculate_win_rate_cycle(signal_df, full_df):
    if len(signal_df) < 2: return 0, 0, 0, 0, 0, []
    signal_df['type'] = np.where(signal_df['buy_signal'].notna(), 'buy', 'sell')
    wins, losses, total_pl = 0, 0, 0.0
    trade_log = []
    signal_df['group'] = (signal_df['type'] != signal_df['type'].shift()).cumsum()
    entry_signals = signal_df.drop_duplicates(subset='group')
    for i in range(len(entry_signals) - 1):
        entry_signal, exit_signal = entry_signals.iloc[i], entry_signals.iloc[i+1]
        entry_price, entry_date = entry_signal['Close'], entry_signal.name
        exit_price, exit_date = exit_signal['Close'], exit_signal.name
        trade_type = "買い" if entry_signal['type'] == 'buy' else "売り"
        trade_period = full_df.loc[entry_date:exit_date]['Close']
        if trade_type == "買い":
            profit = exit_price - entry_price; running_max = trade_period.cummax()
            drawdown = running_max - trade_period; max_drawdown = drawdown.max()
        else: # 売り
            profit = entry_price - exit_price; running_min = trade_period.cummin()
            drawdown = trade_period - running_min; max_drawdown = drawdown.max()
        max_dd_percent = (max_drawdown / entry_price) * 100 if entry_price > 0 else 0
        if profit > 0: wins += 1
        else: losses += 1
        total_pl += profit
        trade_log.append({"取引種別": trade_type, "エントリー日": entry_date.strftime('%Y-%m-%d'), "エントリー価格": f"{entry_price:,.2f}", "決済日": exit_date.strftime('%Y-%m-%d'), "決済価格": f"{exit_price:,.2f}", "損益 (円/株)": f"{profit:+.2f}", "最大DD (円/株)": f"{-max_drawdown:.2f}", "最大DD率 (%)": f"{-max_dd_percent:.2f}%"})
    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    return win_rate, total_trades, wins, losses, total_pl, trade_log
def run_backtest(signal_df, full_df, sl_pct, tp_pct, comm_pct):
    if signal_df.empty: return 0, 0, 0, 0, 0
    signal_df['type'] = np.where(signal_df['buy_signal'].notna(), 'buy', 'sell')
    signal_df['group'] = (signal_df['type'] != signal_df['type'].shift()).cumsum()
    entry_signals = signal_df.drop_duplicates(subset='group')
    wins, losses, total_pl = 0, 0, 0.0
    for i in range(len(entry_signals)):
        entry_signal = entry_signals.iloc[i]; entry_price, entry_date = entry_signal['Close'], entry_signal.name
        next_opposite_signal = entry_signals[entry_signals['group'] > entry_signal['group']].iloc[0] if i < len(entry_signals) - 1 else None
        start_date = entry_date; end_date = next_opposite_signal.name if next_opposite_signal is not None else full_df.index[-1]
        trade_period_df = full_df.loc[start_date:end_date].iloc[1:]
        exit_price = 0; exit_reason = None
        sl_price = entry_price * (1 - sl_pct / 100) if entry_signal['type'] == 'buy' else entry_price * (1 + sl_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100) if entry_signal['type'] == 'buy' else entry_price * (1 - tp_pct / 100)
        for date, row in trade_period_df.iterrows():
            if entry_signal['type'] == 'buy':
                if sl_pct > 0 and row['Low'] <= sl_price: exit_price = sl_price; exit_reason = "sl"; break
                if tp_pct > 0 and row['High'] >= tp_price: exit_price = tp_price; exit_reason = "tp"; break
            elif entry_signal['type'] == 'sell':
                if sl_pct > 0 and row['High'] >= sl_price: exit_price = sl_price; exit_reason = "sl"; break
                if tp_pct > 0 and row['Low'] <= tp_price: exit_price = tp_price; exit_reason = "tp"; break
            if next_opposite_signal is not None and date == next_opposite_signal.name: exit_price = next_opposite_signal['Close']; exit_reason = "signal"; break
        if exit_reason:
            commission = (entry_price * comm_pct / 100) + (exit_price * comm_pct / 100)
            if entry_signal['type'] == 'buy':
                profit = exit_price - entry_price - commission
                if profit > 0: wins += 1
                else: losses += 1
                total_pl += profit
            elif entry_signal['type'] == 'sell':
                profit = entry_price - exit_price - commission
                if profit > 0: wins += 1
                else: losses += 1
                total_pl += profit
    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    return win_rate, total_trades, wins, losses, total_pl

# --- メイン処理 ---
if not user_input:
    st.warning("銘柄コードを入力してください。")
elif jpx_df is None:
    st.error("銘柄リストファイル `jpx_list.xls` が見つかりません。スクリプトと同じフォルダに配置してください。")
else:
    ticker = user_input + ".T" if len(user_input) == 4 and user_input.isdigit() else user_input
    try:
        @st.cache_data
        def get_stock_data(ticker_symbol, period_str):
            return yf.download(ticker_symbol, period=period_str, auto_adjust=True, progress=False)
        df = get_stock_data(ticker, period_options[selected_period])

        if df.empty:
            st.error(f"データを取得できませんでした。")
        else:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            df.columns = [col.capitalize() for col in df.columns]

            # (分析ロジック部分は変更なし)
            df[f'MA{ma_period}'] = df['Close'].rolling(window=ma_period, min_periods=1).mean(); df['Avg_Volume_20'] = df['Volume'].rolling(window=20, min_periods=1).mean(); df['Body'] = abs(df['Close'] - df['Open']); df['Upper_Wick'] = df['High'] - np.maximum(df['Open'], df['Close']); df['Lower_Wick'] = np.minimum(df['Open'], df['Close']) - df['Low']
            if len(df) > 14: df.ta.adx(length=14, append=True)
            else: df[f'ADX_14'] = 0
            buy_score = pd.Series(0, index=df.index); sell_score = pd.Series(0, index=df.index); is_downtrend = df['Close'] < df[f'MA{ma_period}']; is_uptrend = df['Close'] > df[f'MA{ma_period}']; has_long_lower_wick = (df['Lower_Wick'] >= wick_ratio * df['Body']) & (df['Body'] > 0.01); has_long_upper_wick = (df['Upper_Wick'] >= wick_ratio * df['Body']) & (df['Body'] > 0.01); is_high_volume = df['Volume'] >= volume_multiplier * df['Avg_Volume_20']; has_strong_trend = df[f'ADX_14'] > adx_threshold
            buy_score += np.where(is_downtrend & has_long_lower_wick, 3, 0); buy_score += np.where(is_downtrend & is_high_volume, 3, 0); buy_score += np.where(is_downtrend, 2, 0); buy_score += np.where(is_downtrend & has_strong_trend, 2, 0)
            sell_score += np.where(is_uptrend & has_long_upper_wick, 3, 0); sell_score += np.where(is_uptrend & is_high_volume, 3, 0); sell_score += np.where(is_uptrend, 2, 0); sell_score += np.where(is_uptrend & has_strong_trend, 2, 0)
            df['buy_score'] = buy_score; df['sell_score'] = sell_score; df['adx_peak_out'] = (df[f'ADX_14'] > 40) & (df[f'ADX_14'] < df[f'ADX_14'].shift(1)) & (df[f'ADX_14'].shift(1) < df[f'ADX_14'].shift(2))
            df['buy_high'] = np.nan; df['buy_mid'] = np.nan; df['buy_low'] = np.nan; df['sell_high'] = np.nan; df['sell_mid'] = np.nan; df['sell_low'] = np.nan
            df.loc[df['buy_score'] >= 9, 'buy_high'] = df['Low'] * 0.97; df.loc[(df['buy_score'] >= 6) & (df['buy_score'] < 9), 'buy_mid'] = df['Low'] * 0.98; df.loc[(df['buy_score'] >= 4) & (df['buy_score'] < 6), 'buy_low'] = df['Low'] * 0.99
            df.loc[df['sell_score'] >= 9, 'sell_high'] = df['High'] * 1.03; df.loc[(df['sell_score'] >= 6) & (df['sell_score'] < 9), 'sell_mid'] = df['High'] * 1.02; df.loc[(df['sell_score'] >= 4) & (df['sell_score'] < 6), 'sell_low'] = df['High'] * 1.01
            
            # --- 銘柄情報の取得と表示 ---
            company_info = jpx_df[jpx_df['コード'] == user_input]
            if not company_info.empty:
                company_name_display = company_info.iloc[0]['銘柄名']; industry_name = company_info.iloc[0]['33業種区分']
                st.header(f"【{company_name_display} ({ticker})】 - {industry_name}")
            else: st.header(f"【{ticker}】VPA分析")
            
            st.subheader("VPA分析サマリー & 最新株価", divider='rainbow')
            
            # (以降のサマリー表示、バックテスト、チャート描画、履歴、ログ機能は変更なし)
            col_price, col_signal = st.columns((1, 2))
            with col_price:
                if len(df) >= 2:
                    current_price = df['Close'].iloc[-1]; prev_price = df['Close'].iloc[-2]; price_change = current_price - prev_price; pct_change = (price_change / prev_price) * 100 if prev_price > 0 else 0
                    st.metric(label=f"現在値 ({df.index[-1].strftime('%Y-%m-%d')})", value=f"{current_price:,.0f} 円", delta=f"{price_change:+.2f} 円 ({pct_change:+.2f}%)")
                else: st.metric(label="現在値", value=f"{df['Close'].iloc[-1]:,.0f} 円", delta="前日データなし")
            with col_signal:
                all_signals_df = df.dropna(subset=['buy_high', 'buy_mid', 'sell_high', 'sell_mid'], how='all').copy()
                last_signal_row = all_signals_df.iloc[-1] if not all_signals_df.empty else None
                if last_signal_row is not None and last_signal_row['buy_score'] > last_signal_row['sell_score']:
                    signal_price = last_signal_row['Close']; entry_date = last_signal_row.name; current_price = df['Close'].iloc[-1]; price_diff = current_price - signal_price
                    level = '高確率' if pd.notna(last_signal_row['buy_high']) else '中確率'
                    signal_type = f"買い 🟢 ({level})"; pct_diff = (price_diff / signal_price) * 100 if signal_price > 0 else 0
                    st.metric(label=f"最新の買いシグナル ({entry_date.strftime('%Y-%m-%d')} @ {signal_price:,.0f}円)", value=signal_type, delta=f"現在までの含み益: {price_diff:+.2f} 円 ({pct_change:+.2f}%)")
                else: st.metric("最新の買いシグナル", "現在、有効なポジションはありません")
            st.divider()
            
            st.subheader("バックテスト結果", divider='blue')
            df_high_bt = df[(df['buy_high'].notna()) | (df['sell_high'].notna())].copy(); df_high_bt['buy_signal'] = df_high_bt['buy_high']; df_high_bt['sell_signal'] = df_high_bt['sell_high']
            df_mid_up_bt = df[(df['buy_mid'].notna()) | (df['sell_mid'].notna()) | (df['buy_high'].notna()) | (df['sell_high'].notna())].copy(); df_mid_up_bt['buy_signal'] = df_mid_up_bt['buy_mid'].fillna(df_mid_up_bt['buy_high']); df_mid_up_bt['sell_signal'] = df_mid_up_bt['sell_mid'].fillna(df_mid_up_bt['sell_high'])
            df_low_up_bt = df[df['buy_low'].notna() | df['sell_low'].notna() | df['buy_mid'].notna() | df['sell_mid'].notna() | df['buy_high'].notna() | df['sell_high'].notna()].copy(); df_low_up_bt['buy_signal'] = df_low_up_bt['buy_low'].fillna(df_low_up_bt['buy_mid']).fillna(df_low_up_bt['buy_high']); df_low_up_bt['sell_signal'] = df_low_up_bt['sell_low'].fillna(df_low_up_bt['sell_mid']).fillna(df_low_up_bt['sell_high']); level_names = ["高確率のみ", "中確率以上", "低確率以上"]
            tab1, tab2 = st.tabs(["利確・ロスカット適用時の成績", "シグナルサイクルでの成績"]);
            def display_backtest_table(results, level_names):
                cols = st.columns((2, 2, 3, 2, 2)); headers = ["確率レベル", "勝率", "累計損益 (円/株)", "トレード数", "勝ち / 負け"];
                for col, header in zip(cols, headers): col.markdown(f"**{header}**")
                st.divider()
                for level_name, result in zip(level_names, results):
                    wr, tt, w, l, tpl, _ = result if len(result) == 6 else (*result, [])
                    win_color = "#34A853" if wr >= 50 else "#EA4335"; pl_color = "#34A853" if tpl >= 0 else "#EA4335"; cols = st.columns((2, 2, 3, 2, 2))
                    cols[0].markdown(f"<h5>{level_name}</h5>", unsafe_allow_html=True)
                    if tt > 0:
                        cols[1].markdown(f"<h5 style='color:{win_color};'>{wr:.2f}%</h5>", unsafe_allow_html=True); cols[2].markdown(f"<h5 style='color:{pl_color};'>{tpl:+.2f}</h5>", unsafe_allow_html=True)
                        cols[3].markdown(f"<h5>{tt} 回</h5>", unsafe_allow_html=True); cols[4].markdown(f"<h5>{w} 勝 / {l} 敗</h5>", unsafe_allow_html=True)
                    else:
                        for i in range(1, 5): cols[i].markdown("<h5>-</h5>", unsafe_allow_html=True)
            with tab1:
                st.info(f"利確: {take_profit_percent}%、ロスカット: {stop_loss_percent}%、手数料: {commission_percent}% を適用"); results_sl = [run_backtest(df_high_bt, df, stop_loss_percent, take_profit_percent, commission_percent), run_backtest(df_mid_up_bt, df, stop_loss_percent, take_profit_percent, commission_percent), run_backtest(df_low_up_bt, df, stop_loss_percent, take_profit_percent, commission_percent)]; display_backtest_table(results_sl, level_names)
            with tab2:
                st.info("反対シグナル決済（利確・ロスカット・手数料なし）")
                results_cycle = [calculate_win_rate_cycle(df_high_bt, df), calculate_win_rate_cycle(df_mid_up_bt, df), calculate_win_rate_cycle(df_low_up_bt, df)]; display_backtest_table(results_cycle, level_names)
            st.divider()
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=(f'Price Chart for {ticker}', 'Volume'), row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            adx_peak_signals = df[df['adx_peak_out']]; fig.add_trace(go.Scatter(x=adx_peak_signals.index, y=adx_peak_signals['High'] * 1.05, mode='markers', marker=dict(symbol='diamond', color='yellow', size=7, opacity=0.7), name='ADXピークアウト(注意)'), row=1, col=1)
            if show_low_prob:
                fig.add_trace(go.Scatter(x=df.index, y=df['buy_low'], mode='markers', marker=dict(symbol='x', color='grey', size=7), name='低確率(買い)'), row=1, col=1); fig.add_trace(go.Scatter(x=df.index, y=df['sell_low'], mode='markers', marker=dict(symbol='x', color='grey', size=7), name='低確率(売り)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['buy_mid'], mode='markers', marker=dict(symbol='triangle-up', color='gold', size=12), name='中確率(買い)'), row=1, col=1); fig.add_trace(go.Scatter(x=df.index, y=df['sell_mid'], mode='markers', marker=dict(symbol='triangle-down', color='orange', size=12), name='中確率(売り)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['buy_high'], mode='markers', marker=dict(symbol='circle', color='aqua', size=25, opacity=0.4), hoverinfo='none', name='高確率(買い)'), row=1, col=1); fig.add_trace(go.Scatter(x=df.index, y=df['sell_high'], mode='markers', marker=dict(symbol='circle', color='magenta', size=25, opacity=0.4), hoverinfo='none', name='高確率(売り)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['buy_high'], mode='markers', marker=dict(symbol='triangle-up', color='white', size=11), showlegend=False, hoverinfo='none'), row=1, col=1); fig.add_trace(go.Scatter(x=df.index, y=df['sell_high'], mode='markers', marker=dict(symbol='triangle-down', color='white', size=11), showlegend=False, hoverinfo='none'), row=1, col=1)
            fig.update_layout(template="plotly_dark", height=550, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_rangeslider_visible=False, xaxis=dict(title_text=None, rangebreaks=[dict(bounds=["sat", "mon"])]))
            volume_colors = ['#00b386' if row.Close > row.Open else '#ff6347' for index, row in df.iterrows()]; fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=volume_colors), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Avg_Volume_20'], name='出来高平均(20日)', line=dict(color='#ffd700', width=1, dash='dash')), row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("VPAシグナル履歴", divider='blue')
            hist_tab1, hist_tab2, hist_tab3 = st.tabs(["高確率", "中確率", "低確率"])
            all_signals = {'high': [], 'mid': [], 'low': []}; max_score = 10
            for level in ['high', 'mid', 'low']:
                for signal_type in ['buy', 'sell']:
                    col_name = f"{signal_type}_{level}"; prob_text = {'high': '高確率', 'mid': '中確率', 'low': '低確率'}[level]; type_text = '買い 🟢' if signal_type == 'buy' else '売り 🔴'; temp_df = df[df[col_name].notna()].copy()
                    for date, row in temp_df.iterrows():
                        score = int(row[f"{signal_type}_score"]); percentage = (score / max_score) * 100
                        all_signals[level].append({ "日付": date, "シグナル": f"{type_text}", "VPA一致率": f"{percentage:.0f}%", "終値": f"{row['Close']:,.0f} 円" })
            for level, tab in zip(['high', 'mid', 'low'], [hist_tab1, hist_tab2, hist_tab3]):
                with tab:
                    if all_signals[level]:
                        history_df = pd.DataFrame(all_signals[level]).sort_values(by="日付", ascending=False); history_df['日付'] = history_df['日付'].dt.strftime('%Y-%m-%d')
                        st.dataframe(history_df, use_container_width=True, hide_index=True)
                    else: st.info(f"選択した期間に{level}確率のシグナルはありませんでした。")
            
            st.subheader("シグナルサイクル トレード詳細ログ", divider='blue')
            log_level = st.selectbox("表示する確率レベルを選択", options=level_names, key="log_level_select")
            if log_level == "高確率のみ": log_df = df_high_bt
            elif log_level == "中確率以上": log_df = df_mid_up_bt
            else: log_df = df_low_up_bt
            _, _, _, _, _, trade_log = calculate_win_rate_cycle(log_df, df)
            if trade_log:
                log_display_df = pd.DataFrame(trade_log)
                html = "<table><tr>"
                for col in log_display_df.columns: html += f"<th>{col}</th>"
                html += "</tr>"
                for index, row in log_display_df.iterrows():
                    profit_val = float(row['損益 (円/株)']); color = "#87CEEB" if profit_val > 0 else "#F08080" if profit_val < 0 else "white"
                    html += f"<tr style='color: {color};'>"
                    for col in log_display_df.columns: html += f"<td>{row[col]}</td>"
                    html += "</tr>"
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
            else: st.info("選択したレベルでは完了したトレードがありません。")

    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
        st.info("解決しない場合、ライブラリのバージョン互換性の問題が考えられます。")