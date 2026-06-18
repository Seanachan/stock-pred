"""The ~43 new universe names added in v6 (sourced from TWSE 0050 + mid-cap
0051 holdings from wantgoo.com as of 2026-06-12, verified TWSE/TPEX-only
against stock_symbol_map.json, deduped against the existing 46). (code, name)
pairs; codes append to stock_ids in order. See
docs/superpowers/specs/2026-06-14-v6-universe-expansion-design.md."""

NEW_STOCKS: list[tuple[str, str]] = [
    # 0050 large-caps not already in the 46 (by index weight order)
    ("3711", "日月光投控"),  # ASE Tech - semis packaging
    ("2383", "台光電"),      # Taiwan Optical Elect - PCB/laminates
    ("2327", "國巨"),        # Yageo - passives
    ("3037", "欣興"),        # Unimicron - PCB
    ("2345", "智邦"),        # Accton - networking
    ("2360", "致茂"),        # Chroma ATE - test equip
    ("3017", "奇鋐"),        # Auras Tech - thermal mgmt
    ("2357", "華碩"),        # ASUS - PCs/motherboards
    ("6669", "緯穎"),        # Wiwynn - cloud servers
    ("3653", "健策"),        # GCHEM - thermal interface
    ("2883", "凱基金"),      # KGI Financial - financials
    ("2368", "金像電"),      # Gold Circuit Elect - PCB
    ("2449", "京元電子"),    # King Yuan Elect - test/packaging
    ("2344", "華邦電"),      # Winbond - DRAM/NOR flash
    ("2301", "光寶科"),      # Lite-On - electronics
    ("2408", "南亞科"),      # Nanya Technology - DRAM
    ("3661", "世芯-KY"),     # Alchip - IC design/ASIC
    ("2059", "川湖"),        # Chuan Lake - server rails/furniture
    ("2395", "研華"),        # Advantech - industrial PC
    # 0051 liquid mid-caps (by weight/liquidity)
    ("2379", "瑞昱"),        # Realtek - IC design
    ("3443", "創意"),        # Global Unichip - IC design
    ("5871", "中租-KY"),     # Chailease - leasing
    ("4938", "和碩"),        # Pegatron - EMS
    ("2324", "仁寶"),        # Compal - laptops
    ("2356", "英業達"),      # Inventec - servers
    ("2347", "聯強"),        # Synnex TW - distribution
    ("1590", "亞德客-KY"),   # Airtac - pneumatic components
    ("5876", "上海商銀"),    # Shanghai Commercial Bank
    # NOTE: 6446 藥華藥 (IPO 2024-01, only ~573 bars) dropped from the universe —
    # build_tensors intersects dates, so its short history collapsed the aligned
    # training window (T 440->222). All other names cover the 2020+ window.
    ("3036", "文曄"),        # WPG Holdings - distribution
    ("2409", "友達"),        # AUO - display panels
    ("1402", "遠東新"),      # Far Eastern New Century - textiles
    ("2353", "宏碁"),        # Acer - PCs
    ("9904", "寶成"),        # Pou Chen - footwear OEM
    ("2049", "上銀"),        # Hiwin - linear motion
    ("1476", "儒鴻"),        # Eclat Textile - functional fabric
    ("2105", "正新"),        # Cheng Shin Rubber - tires
    ("2377", "微星"),        # MSI - gaming/laptops
    ("6285", "啟碁"),        # Wistron NeWeb - networking
    ("5269", "祥碩"),        # ASMedia - USB/IO controllers
    ("2633", "台灣高鐵"),    # THSR - transport
    ("2610", "華航"),        # China Airlines - aviation
    ("2801", "彰銀"),        # Chang Hwa Bank - banking
]
