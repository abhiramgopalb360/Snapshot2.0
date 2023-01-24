def epsilon_preprocess(df):

    # INDIVIDUAL_EXACT_AGE1 -> SS_INDIVIDUAL_EXACT_AGE1
    def age_binning(x):
        if x >= 18 and x <= 24:
            return 1
        elif x >= 25 and x <= 34:
            return 2
        elif x >= 35 and x <= 44:
            return 3
        elif x >= 45 and x <= 54:
            return 4
        elif x >= 55 and x <= 64:
            return 5
        elif x >= 65 and x <= 74:
            return 6
        elif x >= 75 and x <= 150:
            return 7
        else:
            return 99

    try:
        df["SS_INDIVIDUAL_EXACT_AGE1"] = df["INDIVIDUAL_EXACT_AGE1"].apply(age_binning)
        # df = df.drop('INDIVIDUAL_EXACT_AGE1',axis=1)
    except Exception:
        pass

    # gender 1-> SS_GENDER1
    def num_process2(x):
        if x != 1 and x != 2:
            x = 99
        return x

    try:
        df["SS_GENDER1"] = df["GENDER1"].apply(num_process2)
        # df = df.drop('GENDER1',axis=1)
    except Exception:
        pass
    # ADV_IND_MARITAL_STATUS1 -> SS_ADV_IND_MARITAL_STATUS1
    df["SS_ADV_IND_MARITAL_STATUS1"] = df["ADV_IND_MARITAL_STATUS1"].apply(num_process2)
    # df = df.drop('ADV_IND_MARITAL_STATUS1',axis=1)

    # ETHNIC_GROUP1 -> SS_ETHNIC_GROUP1
    def letter_process_1(x):
        if x not in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "O", "Y"]:
            x = "Z"
        return x

    try:
        df["SS_ETHNIC_GROUP1"] = df["ETHNIC_GROUP1"].apply(letter_process_1)
        # df = df.drop('ETHNIC_GROUP1',axis=1)
    except Exception:
        pass

    # RELIGION1 -> SS_RELIGION1
    def letter_process_2(x):
        if x not in ["B", "C", "G", "H", "I", "J", "K", "L", "M", "O", "P", "S"]:
            x = "Z"
        return x

    try:
        df["SS_RELIGION1"] = df["RELIGION1"].apply(letter_process_2)
        # df = df.drop('RELIGION1',axis=1)
    except Exception:
        pass

    # POLITICAL_PARTY_INDIVIDUAL_1 ->SS_POLITICAL_PARTY_INDIVIDUAL_1
    def letter_process_3(x):
        if x not in ["D", "R", "I"]:
            x = "Z"
        return x

    try:
        df["SS_POLITICAL_PARTY_INDIVIDUAL_1"] = df["POLITICAL_PARTY_INDIVIDUAL_1"].apply(letter_process_2)
        # df = df.drop('POLITICAL_PARTY_INDIVIDUAL_1',axis=1)
    except Exception:
        pass

    # ADV_HH_AGE_CODE_ENH->SS_ADV_HH_AGE_CODE_ENH
    def num_process7(x):
        if x not in [1, 2, 3, 4, 5, 6, 7]:
            x = 99
        return x

    try:
        df["SS_ADV_HH_AGE_CODE_ENH"] = df["ADV_HH_AGE_CODE_ENH"].apply(num_process7)
        # df = df.drop('ADV_HH_AGE_CODE_ENH',axis=1)
    except Exception:
        pass
    #  AGE_18_24_SPEC_ENH,AGE_25_34_SPEC_ENH,AGE_35_44_SPEC_ENH,AGE_45_54_SPEC_ENH,
    # AGE_55_64_SPEC_ENH,AGE_65_74_SPEC_ENH,PREZ_ADULT_75_ABOVE_ENH    ->  SS_ADULT_AGE_BAND_HH
    # leave for later

    # ADV_HH_MARITAL_STATUS ->SS_ADV_HH_MARITAL_STATUS
    def num_process3(x):
        if x not in [1, 2, 3]:
            x = 99
        return x

    try:
        df["SS_ADV_HH_MARITAL_STATUS"] = df["ADV_HH_MARITAL_STATUS"].apply(num_process3)
        # df = df.drop('ADV_HH_MARITAL_STATUS',axis=1)
    except Exception:
        pass

    # ADV_LENGTH_RESIDENCE ->SS_ADV_LENGTH_RESIDENCE
    def num_process8(x):
        if x not in [1, 2, 3, 4, 5, 6, 7, 8]:
            x = 99
        return x

    try:
        df["SS_ADV_LENGTH_RESIDENCE"] = df["ADV_LENGTH_RESIDENCE"].apply(num_process8)
        # df = df.drop('ADV_LENGTH_RESIDENCE',axis=1)
    except Exception:
        pass

    # ADV_HH_SIZE_ENH -> SS_ADV_HH_SIZE_ENH
    def num_process9(x):
        if x not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            x = 99
        return x

    try:
        df["SS_ADV_HH_SIZE_ENH"] = df["ADV_HH_SIZE_ENH"].apply(num_process9)
        # df = df.drop('ADV_HH_SIZE_ENH',axis=1)
    except Exception:
        pass

    # ADV_NUM_ADULTS->SS_ADV_NUM_ADULTS
    def num_process5(x):
        if x not in [1, 2, 3, 4, 5]:
            x = 99
        return x

    try:
        df["SS_ADV_NUM_ADULTS"] = df["ADV_NUM_ADULTS"].apply(num_process5)
        # df = df.drop('ADV_NUM_ADULTS',axis=1)
    except Exception:
        pass

    # ADV_PREZ_CHILDREN_ENH ->SS_ADV_PREZ_CHILDREN_ENH
    def num_process1(x):
        if x not in [0, 1]:
            x = 99
        return x

    try:
        df["SS_ADV_PREZ_CHILDREN_ENH"] = df["ADV_PREZ_CHILDREN_ENH"].apply(num_process1)
        # df = df.drop('ADV_PREZ_CHILDREN_ENH',axis=1)
    except Exception:
        pass
    # NUM_CHILDREN_HH_ENH ->SS_NUM_CHILDREN_HH_ENH
    try:
        df["SS_NUM_CHILDREN_HH_ENH"] = df["NUM_CHILDREN_HH_ENH"].apply(num_process9)
        # df = df.drop('NUM_CHILDREN_HH_ENH',axis=1)
    except Exception:
        pass
    # NUM_GENERATIONS_HH_ENH ->SS_NUM_GENERATIONS_HH_ENH
    try:
        df["SS_NUM_GENERATIONS_HH_ENH"] = df["NUM_GENERATIONS_HH_ENH"].apply(num_process5)
        # df = df.drop('NUM_GENERATIONS_HH_ENH',axis=1)
    except Exception:
        pass
    # ADV_HH_EDU_ENH ->SS_ADV_HH_EDU_ENH
    try:
        df["SS_ADV_HH_EDU_ENH"] = df["ADV_HH_EDU_ENH"].apply(num_process5)
        # df = df.drop('ADV_HH_EDU_ENH',axis=1)
    except Exception:
        pass
    # ETHNIC_GROUP_CODE_HOUSEHOLD->SS_ETHNIC_GROUP_CODE_HOUSEHOLD
    try:
        df["ETHNIC_GROUP_CODE_HOUSEHOLD"] = df["ETHNIC_GROUP_CODE_HOUSEHOLD"].apply(letter_process_1)
        # df = df.drop('ETHNIC_GROUP_CODE_HOUSEHOLD',axis=1)
    except Exception:
        pass

    # POLITICAL_PARTY_HH->SS_POLITICAL_PARTY_HH
    try:
        df["SS_POLITICAL_PARTY_HH"] = df["POLITICAL_PARTY_HH"].apply(num_process7)
        # df = df.drop('POLITICAL_PARTY_HH',axis=1)
    except Exception:
        pass

    # ADV_DWELLING_TYP->SS_ADV_DWELLING_TYP
    def num_process6(x):
        if x not in [1, 2, 3, 4, 5, 6]:
            x = 99
        return x

    try:
        df["SS_ADV_DWELLING_TYP"] = df["ADV_DWELLING_TYP"].apply(num_process6)
        # df = df.drop('ADV_DWELLING_TYP',axis=1)
    except Exception:
        pass

    # OCCUPATION->SS_OCCUPATION
    def num_process20(x):
        if x not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            x = 99
        return x

    try:
        df["SS_OCCUPATION"] = df["OCCUPATION"].apply(num_process20)
        # df = df.drop('OCCUPATION',axis=1)
    except Exception:
        pass

    # TGT_PRE_MOVER_20_MODEL ->SS_TGT_PRE_MOVER_20_MODEL
    def rank_process(x):
        if x >= 1 and x <= 5:
            x = x
        elif x >= 6 and x <= 20:
            x = 6
        elif x >= 21 and x <= 40:
            x = 7
        elif x >= 41 and x <= 60:
            x = 8
        elif x >= 61 and x <= 80:
            x = 9
        elif x >= 81 and x <= 99:
            x = 10
        else:
            x = 99
        return x

    try:
        df["SS_TGT_PRE_MOVER_20_MODEL"] = df["TGT_PRE_MOVER_20_MODEL"].apply(num_process20)
        # df = df.drop('TGT_PRE_MOVER_20_MODEL',axis=1)
    except Exception:
        pass

    # STATE->SS_STATE
    def state(x):
        if x not in (
            ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS"]
            + ["KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV"]
            + ["NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]
        ):
            return "ZZ"

    try:
        df["SS_STATE"] = df["STATE"].apply(state)
        # df = df.drop('STATE',axis=1)
    except Exception:
        pass

    # ADV_TGT_INCOME_30 -> SS_ADV_TGT_INCOME_30
    def income(x):
        if x not in ["4", "6", "A", "9", "5", "8", "1", "7", "2", "3", "B", "C", "D"]:
            x = "Z"
        return x

    try:
        df["SS_ADV_TGT_INCOME_30"] = df["ADV_TGT_INCOME_30"].apply(income)
        # df = df.drop('ADV_TGT_INCOME_30',axis=1)
    except Exception:
        pass

    # TARGET_NET_WORTH_4_0 ->SS_TARGET_NET_WORTH_4_0
    def net_worth(x):
        if x not in ["0", "2", "1", "5", "9", "B", "8", "6", "7", "4", "3", "A"]:
            x = "Z"
        return x

    try:
        df["SS_TARGET_NET_WORTH_4_0"] = df["TARGET_NET_WORTH_4_0"].apply(net_worth)
        # df = df.drop('TARGET_NET_WORTH_4_0',axis=1)
    except Exception:
        pass
    # SHORT_TERM_LIABILITY->SS_SHORT_TERM_LIABILITY
    try:
        df["SS_SHORT_TERM_LIABILITY"] = df["SHORT_TERM_LIABILITY"].apply(num_process9)
    except Exception:
        pass
        # df = df.drop('SHORT_TERM_LIABILITY',axis=1)
    # WEALTH_RESOURCES->SS_WEALTH_RESOURCES

    def num_process10(x):
        if x not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            x = 99
        return x

    try:
        df["SS_WEALTH_RESOURCES"] = df["WEALTH_RESOURCES"].apply(num_process10)
        # df = df.drop('WEALTH_RESOURCES',axis=1)
    except Exception:
        pass

    # INVESTMENT_RESOURCES ->SS_INVESTMENT_RESOURCES
    def num_process12(x):
        if x not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            x = 99
        return x

    try:
        df["SS_INVESTMENT_RESOURCES"] = df["INVESTMENT_RESOURCES"].apply(num_process12)
        # df = df.drop('INVESTMENT_RESOURCES',axis=1)
    except Exception:
        pass

    # LIQUID_RESOURCES ->SS_LIQUID_RESOURCES
    def num_process14(x):
        if x not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
            x = 99
        return x

    try:
        df["SS_LIQUID_RESOURCES"] = df["LIQUID_RESOURCES"].apply(num_process14)
    except Exception:
        pass

    # MERITSCORE,TARGET_VALUESCORE_20_ALL_MARKETERS,TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS
    def letter_process4(x):
        if x[0] in ["A", "B", "C", "D", "E"]:
            x = x[0]
        else:
            x = "Z"
        return x

    for i in ["MERITSCORE", "TARGET_VALUESCORE_20_ALL_MARKETERS", "TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS"]:
        try:
            df["SS_" + i] = df[i].apply(letter_process4)
        except Exception:
            pass

    # EAI_PERFORMANCE_RANK
    def perf_rank(x):
        if x == "1H":
            x = 1
        elif x == "1G":
            x = 2
        elif x == "2H":
            x = 3
        elif x == "2G":
            x = 4
        elif x == "3H":
            x = 5
        elif x == "3G":
            x = 6
        elif x == "4H":
            x = 7
        elif x == "4G":
            x = 8
        else:
            x = 99
        return x

    try:
        df["SS_EAI_PERFORMANCE_RANK"] = df["EAI_PERFORMANCE_RANK"].apply(perf_rank)
    except Exception:
        pass
    # 'CREDIT_ACTIVE','OCREDIT_FIN_SERVICES_INSTALL','OCREDIT_FIN_SERVICES_BANKING','OCREDIT_HOME_MORTG'
    # ,'OCREDIT_AUTO_LOANS','OCREDIT_EDU_STUDENT_LOANS','OCREDIT_LEASING',,'OCREDIT_FIN_SERVICES_INSURANCE'
    # skip

    # ADV_HOME_OWNER ->SS_ADV_HOME_OWNER
    def num_process4(x):
        if x not in [1, 2, 3, 4]:
            x = 99
        return x

    try:
        df["SS_ADV_HOME_OWNER"] = df["ADV_HOME_OWNER"].apply(num_process4)
    except Exception:
        pass

    # HOME_VALUE_TIERS->SS_HOME_VALUE_TIERS
    def value_tier(x):
        if x >= 1 and x <= 100:
            x = 1
        elif x >= 101 and x <= 200:
            x = 2
        elif x >= 201 and x <= 300:
            x = 3
        elif x >= 301 and x <= 400:
            x = 4
        elif x >= 401 and x <= 500:
            x = 5
        elif x >= 501 and x <= 600:
            x = 6
        elif x >= 601 and x <= 700:
            x = 7
        elif x >= 701 and x <= 800:
            x = 8
        elif x >= 801 and x <= 900:
            x = 9
        elif x >= 901 and x <= 9999:
            x = 10
        else:
            x = 99
        return x

    try:
        df["SS_HOME_VALUE_TIERS"] = df["HOME_VALUE_TIERS"].apply(value_tier)
        # df = df.drop(i,axis=1)
    except Exception:
        pass

    # YEAR_HOME_BUILT->SS_YEAR_HOME_BUILT
    def year_categorize(x):
        if x >= 0 and x <= 1:
            x = 1
        elif x >= 2 and x <= 5:
            x = 2
        elif x >= 6 and x <= 10:
            x = 3
        elif x >= 11 and x <= 20:
            x = 4
        elif x >= 21 and x <= 30:
            x = 5
        elif x >= 31 and x <= 40:
            x = 6
        elif x >= 41 and x <= 50:
            x = 7
        elif x >= 51 and x <= 60:
            x = 8
        elif x >= 61 and x <= 70:
            x = 9
        elif x >= 71:
            x = 10
        else:
            x = 99
        return x

    try:
        df["SS_YEAR_HOME_BUILT"] = df["YEAR_HOME_BUILT"].apply(year_categorize)
    except Exception:
        pass

    # MORTG_LIABILITY
    def morgage_categorize(x):
        if x >= 1 and x < 25:
            x = 1
        elif x >= 25 and x < 50:
            x = 2
        elif x >= 50 and x < 100:
            x = 3
        elif x >= 100 and x < 200:
            x = 4
        elif x >= 200 and x < 300:
            x = 5
        elif x >= 300 and x < 400:
            x = 6
        elif x >= 400 and x < 500:
            x = 7
        elif x >= 500 and x < 600:
            x = 8
        elif x >= 600:
            x = 10
        else:
            x = 99
        return x

    try:
        df["SS_MORTG_LIABILITY"] = df["MORTG_LIABILITY"].apply(morgage_categorize)
    except Exception:
        pass

    # CURRENT_LOAN_TO_VALUE
    def loan_value(x):
        if x >= 1 and x < 30:
            x = 1
        elif x >= 30 and x < 40:
            x = 2
        elif x >= 40 and x < 50:
            x = 3
        elif x >= 50 and x < 60:
            x = 4
        elif x >= 60 and x < 70:
            x = 5
        elif x >= 70 and x < 80:
            x = 6
        elif x >= 80 and x < 90:
            x = 7
        elif x >= 90 and x < 100:
            x = 8
        elif x >= 100 and x < 110:
            x = 9
        elif x >= 110:
            x = 10
        else:
            x = 99
        return x

    try:
        df["SS_CURRENT_LOAN_TO_VALUE"] = df["CURRENT_LOAN_TO_VALUE"].apply(loan_value)
    except Exception:
        pass

    # AVA_HOME_EQUITY_IN_K
    def h_equity_categorize(x):
        if x == 0:
            x = 1
        elif x > 0 and x < 10:
            x = 2
        elif x >= 10 and x < 50:
            x = 3
        elif x >= 50 and x < 80:
            x = 4
        elif x >= 80 and x < 100:
            x = 5
        elif x >= 100 and x < 150:
            x = 6
        elif x >= 150 and x < 200:
            x = 7
        elif x >= 200 and x < 250:
            x = 8
        elif x >= 300 and x < 300:
            x = 9
        elif x >= 300:
            x = 10
        else:
            x = 99
        return x

    try:
        df["SS_AVA_HOME_EQUITY_IN_K"] = df["AVA_HOME_EQUITY_IN_K"].apply(h_equity_categorize)
    except Exception:
        pass

    # HOME_SALE_PRICE_IN_K
    def h_price_categorize(x):
        if x >= 1 and x < 100:
            x = 1
        elif x > 100 and x < 200:
            x = 2
        elif x >= 200 and x < 300:
            x = 3
        elif x >= 300 and x < 400:
            x = 4
        elif x >= 400 and x < 500:
            x = 5
        elif x >= 500 and x < 600:
            x = 6
        elif x >= 600 and x < 700:
            x = 7
        elif x >= 700 and x < 800:
            x = 8
        elif x >= 800 and x < 900:
            x = 9
        elif x >= 900:
            x = 10
        else:
            x = 99
        return x

    try:
        df["SS_HOME_SALE_PRICE_IN_K"] = df["HOME_SALE_PRICE_IN_K"].apply(h_price_categorize)
    except Exception:
        pass

    # LIVING_AREA_SQ_FTG_RANGE
    def area(x):
        if x not in ["C", "E", "F", "G", "B", "D", "H", "I", "J", "K", "A", "L", "M", "N"]:
            x = "Z"
        return x

    try:
        df["SS_LIVING_AREA_SQ_FTG_RANGE"] = df["LIVING_AREA_SQ_FTG_RANGE"].apply(area)
    except Exception:
        pass

    # EXTERIOR_WALL_TYP
    def wall_type(x):
        if x in ["A", "D", "G", "H", "R", "S", "T"]:
            x = x
        elif x in ["B", "C", "E", "F", "I", "J", "K", "L", "M", "N", "O", "P", "Q"]:
            x = "Y"
        else:
            x = "Z"
        return x

    try:
        df["SS_EXTERIOR_WALL_TYP"] = df["EXTERIOR_WALL_TYP"].apply(wall_type)
    except Exception:
        pass

    # FUEL_CODE
    def fuel_code(x):
        if x in ["E", "G", "O", "S"]:
            x = x
        elif x in ["C", "U"]:
            x = "Y"
        else:
            x = "Z"
        return x

    try:
        df["SS_FUEL_CODE"] = df["FUEL_CODEP"].apply(fuel_code)
    except Exception:
        pass

    # HOME_HEAT_SOURCE
    def heat_source(x):
        if x in ["A", "B", "D", "F", "H", "I", "K"]:
            x = x
        elif x in ["C", "E", "G", "J", "L", "M", "N"]:
            x = "Y"
        else:
            x = "Z"
        return x

    try:
        df["SS_HOME_HEAT_SOURCE"] = df["HOME_HEAT_SOURCE"].apply(heat_source)
    except Exception:
        pass

    # ROOF_COVER_TYP
    def roof(x):
        if x not in ["A", "B", "C", "D", "F", "E", "G", "H", "I", "J", "K", "L", "M"]:
            x = "Z"
        return x

    try:
        df["SS_ROOF_COVER_TYP"] = df["ROOF_COVER_TYP"].apply(roof)
    except Exception:
        pass
    # 55,56,57,58,59,60,...66, SKIP

    # NUMBER_OF_VEHICLES_IN_HOUSEHOLD
    def num_vehicles(x):
        if x in [1, 2, 3, 4, 5, 6]:
            x = x
        elif x in [7, 8, 9]:
            x = 7
        else:
            x = 9
        return x

    try:
        df["SS_NUMBER_OF_VEHICLES_IN_HOUSEHOLD"] = df["NUMBER_OF_VEHICLES_IN_HOUSEHOLD"].apply(num_vehicles)
    except Exception:
        pass

    # NUMBER_OF_CARS_IN_HOUSEHOLD,NUMBER_OF_TRUCKS_IN_HOUSEHOLD
    def num_cars_trucks(x):
        if x in [1, 2, 3, 4]:
            x = x
        elif x in [5, 6, 7, 8]:
            x = 5
        else:
            x = 9
        return x

    try:
        df["SS_NUMBER_OF_VEHICLES_IN_HOUSEHOLD"] = df["NUMBER_OF_VEHICLES_IN_HOUSEHOLD"].apply(num_cars_trucks)
    except Exception:
        pass
    try:
        df["SS_NUMBER_OF_TRUCKS_IN_HOUSEHOLD"] = df["NUMBER_OF_TRUCKS_IN_HOUSEHOLD"].apply(num_cars_trucks)
    except Exception:
        pass
    # 71 skip
    # LIKELY_TO_BUY_DOMESTIC_VEHICLE,LIKELY_TO_BUY_IMPORT_VEHICLE,

    # LIKELY_TO_BUY_NEW_VEHICLE,LIKELY_TO_BUY_USED_VEHICLE
    def likely_buy(x):
        if x >= 1 and x <= 5:
            x = x
        elif x >= 6 and x <= 20:
            x = 6
        elif x >= 21 and x <= 40:
            x = 7
        elif x >= 41 and x <= 60:
            x = 8
        elif x >= 61 and x <= 80:
            x = 9
        elif x >= 81 and x <= 99:
            x = 10
        else:
            x = 99
        return x

    for i in ["LIKELY_TO_BUY_DOMESTIC_VEHICLE", "LIKELY_TO_BUY_IMPORT_VEHICLE", "LIKELY_TO_BUY_NEW_VEHICLE", "LIKELY_TO_BUY_USED_VEHICLE"]:
        try:
            df["SS_" + i] = df[i].apply(likely_buy)
        except Exception:
            pass
    # ACT_AVG_DOLLARS_QUINT,ACT_TOT_OFFLINE_DOLLARS_QUINT,ACT_TOT_ONLINE_DOLLARS_QUINT,ACT_NUM_OFFLINE_PURCHASE_QUINT,
    # ACT_NUM_ONLINE_PURCHASE_QUINT,CHANNEL_PREF_RT_CATALOG_QUINT,CHANNEL_PREF_RT_ONLINE_QUINT,CHANNEL_PREF_RT_ONLINE_QUINT
    # HH_PURCHASE_CHANNEL_INT,HH_PURCHASE_CHANNEL_MO,CLUB_CONTINUITY_BUYER,PAYMENT_METHOD_CASH,PAYMENT_METHOD_CC
    # BEAUTY_SPA_QUINT,CHILDREN_QUINT,FASHION_ACC_BEAUTY_QUINT,MAGAZINES_QUINT,SPEC_FOOD_GIFT_QUINT,SPORTS_OUTDOOR_QUINT,

    # DS_TIME_FOR_TEACH_KIDS_QUINTILE,DS_TRENDSETTERS_QUINTILE
    def quint(x):
        if x not in [1, 2, 3, 4, 5]:
            x = 9
        return x

    def bi(x):
        if x != "Y":
            x = "Z"
        return x

    for i in [
        "ACT_AVG_DOLLARS_QUINT",
        "ACT_TOT_OFFLINE_DOLLARS_QUINT",
        "ACT_TOT_ONLINE_DOLLARS_QUINT",
        "ACT_NUM_OFFLINE_PURCHASE_QUINT",
        "ACT_NUM_ONLINE_PURCHASE_QUINT",
        "CHANNEL_PREF_RT_CATALOG_QUINT",
        "CHANNEL_PREF_RT_ONLINE_QUINT",
        "CHANNEL_PREF_RT_ONLINE_QUINT",
    ]:
        try:
            df["SS_" + i] = df[i].apply(quint)
        except Exception:
            pass
    for i in ["HH_PURCHASE_CHANNEL_INT", "HH_PURCHASE_CHANNEL_MO", "CLUB_CONTINUITY_BUYER", "PAYMENT_METHOD_CASH", "PAYMENT_METHOD_CC"]:
        try:
            df["SS_" + i] = df[i].apply(bi)
        except Exception:
            pass
    for i in [
        "BEAUTY_SPA_QUINT",
        "CHILDREN_QUINT",
        "FASHION_ACC_BEAUTY_QUINT",
        "MAGAZINES_QUINT",
        "SPEC_FOOD_GIFT_QUINT",
        "SPORTS_OUTDOOR_QUINT",
        "DS_TIME_FOR_TEACH_KIDS_QUINTILE",
        "DS_TRENDSETTERS_QUINTILE",
    ]:
        try:
            df["SS_" + i] = df[i].apply(quint)
        except Exception:
            pass

    # NICHES_40
    def niches_40(x):
        if x[0] in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
            x = x[0]
        else:
            x = "ZZ"
        return x

    try:
        df["SS_NICHES_40"] = df["NICHES_40"].apply(niches_40)
    except Exception:
        pass

    return df
