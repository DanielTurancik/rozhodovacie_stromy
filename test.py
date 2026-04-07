import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

with st.expander("Zobraziť animáciu rozhodovacieho stromu"):
    st.video("animacia.mp4")


dataset_choice = st.session_state.get("dataset", "Úver")
col1, col2 = st.columns(2)
with col1:
    if st.button("Úver", use_container_width=True, type="primary" if dataset_choice == "Úver" else "secondary"):
        st.session_state.dataset = "Úver"
        st.rerun()
with col2:
    if st.button("Titanic", use_container_width=True, type="primary" if dataset_choice == "Titanic" else "secondary"):
        st.session_state.dataset = "Titanic"
        st.rerun()



if dataset_choice == "Úver":
    st.title("Rozhodovací strom - úver")

    train_df = pd.read_csv("testovacia_mnozina_uvery.csv")
    train_df = train_df.drop(columns=["Klient"])

    st.subheader("Tréningová sada")
    # editovacia tabulka -> pridavanie, mazanie, atd...
    edited_train = st.data_editor(
            train_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
            "Príjem":st.column_config.SelectboxColumn("Príjem",options=["Vysoký","Nízky"]),
            "Konto":st.column_config.SelectboxColumn("Konto", options=["Vysoké","Stredné","Nízke"]),
            "Pohlavie":st.column_config.SelectboxColumn("Pohlavie",options=["Muž","Žena"]),
            "Nezamestnaný":st.column_config.SelectboxColumn("Nezamestnaný",options=["Áno","Nie"]),
            "Úver":st.column_config.SelectboxColumn("Úver",options=["Áno","Nie"]),
            }
        )
    st.subheader("Testovacia sada")
    empty_test = pd.DataFrame({
        "Príjem":["Vysoký","Nízky","Vysoký","Nízky","Nízky"],
        "Konto":["Nízke","Vysoké", "Stredné","Nízke","Stredné"],
        "Pohlavie":["Muž","Žena","Žena","Muž", "Žena"],
        "Nezamestnaný":["Nie","Áno","Nie","Áno","Nie"],
        "Úver":["Áno","Nie", "Áno","Nie","Áno"],
    })

    edited_test = st.data_editor(
        empty_test,
        num_rows="dynamic",
        use_container_width=True,
        key="test",
        column_config={
            "Príjem":st.column_config.SelectboxColumn("Príjem",options=["Vysoký","Nízky"]),
            "Konto":st.column_config.SelectboxColumn("Konto", options=["Vysoké","Stredné","Nízke"]),
            "Pohlavie":st.column_config.SelectboxColumn("Pohlavie",options=["Muž","Žena"]),
            "Nezamestnaný":st.column_config.SelectboxColumn("Nezamestnaný",options=["Áno","Nie"]),
            "Úver":st.column_config.SelectboxColumn("Úver",options=["Áno","Nie"]),
        }
    )

    mapa_prijem = {"Vysoký": 1, "Nízky": 0}
    mapa_konto = {"Vysoké": 2, "Stredné": 1, "Nízke": 0}
    mapa_pohlavie = {"Muž": 1, "Žena": 0}
    mapa_nezamestnany =  {"Áno": 1, "Nie": 0}
    mapa_uver = {"Áno": 1, "Nie": 0}

    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        max_depth = st.slider("Maximálna hĺbka stromu", min_value=1, max_value=5, value=2)
    with col_2:
        criterion = st.selectbox("Kritérium", ["gini", "entropy"])
    with col_3:
        min_samples_split = st.slider("Minimálne rozdelenie uzla", 2, 10, 2)


    encrypted_train = edited_train.copy()
    encrypted_train["Príjem"] = encrypted_train["Príjem"].map(mapa_prijem)
    encrypted_train["Konto"] = encrypted_train["Konto"].map(mapa_konto)
    encrypted_train["Pohlavie"] = encrypted_train["Pohlavie"].map(mapa_pohlavie)
    encrypted_train["Nezamestnaný"] = encrypted_train["Nezamestnaný"].map(mapa_nezamestnany)
    encrypted_train["Úver"] = encrypted_train["Úver"].map(mapa_uver)

    encrypted_test = edited_test.copy()
    encrypted_test["Príjem"] = encrypted_test["Príjem"].map(mapa_prijem)
    encrypted_test["Konto"] = encrypted_test["Konto"].map(mapa_konto)
    encrypted_test["Pohlavie"] = encrypted_test["Pohlavie"].map(mapa_pohlavie)
    encrypted_test["Nezamestnaný"] = encrypted_test["Nezamestnaný"].map(mapa_nezamestnany)
    encrypted_test["Úver"] = encrypted_test["Úver"].map(mapa_uver)
    if encrypted_train.isnull().any().any() or encrypted_test.isnull().any().any():
        st.warning("Tabuľka obsahuje prázdne hodnoty")
    else:
        with st.expander("Kódovanie hodnôt"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("**Príjem**")
                st.success("Vysoký = 1")
                st.error("Nízky = 0")
            with col2:
                st.markdown("**Konto**")
                st.success("Vysoké = 2")
                st.warning("Stredné = 1")
                st.error("Nízke = 0")
            with col3:
                st.markdown("**Pohlavie**")
                st.info("Muž = 1")
                st.info("Žena = 0")
            with col4:
                st.markdown("**Nezamestnaný**")
                st.error("Áno = 1")
                st.success("Nie = 0")
        features = ["Príjem", "Konto", "Pohlavie", "Nezamestnaný"]

        X_train = encrypted_train[features].values
        y_train = encrypted_train["Úver"].values
        X_test = encrypted_test[features].values
        y_test = encrypted_test["Úver"].values

        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        clf.fit(X_train, y_train)

        # Distribúcia tried
        _, counts = np.unique(y_train, return_counts=True)
        probs = counts / len(y_train)
        st.caption(
            f"Tréningová sada obsahuje {counts[0]} zamietnutých úverov (Nie) "
            f"a {counts[1]} schválených úverov (Áno) "
            f"z celkových {len(y_train)} záznamov."
        )

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)



        st.success(f"Presnosť: {acc*100:.2f}%       Hĺbka: {clf.get_depth()}      Uzlov: {clf.tree_.node_count}")


        result_df = edited_test.copy()
        result_df["Predpoveď"] = ["Áno" if pred == 1 else "Nie" for pred in y_pred]
        result_df["Správne?"] = ["Áno" if pred == answ else "Nie" for pred, answ in zip(y_pred, y_test)]
        st.dataframe(result_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(14, 6))
        plot_tree(clf, feature_names=features, class_names=["Nie", "Áno"],
                  filled=True, rounded=True, impurity=True, ax=ax)
        st.pyplot(fig)
        st.write(f"Použitá hĺbka: {max_depth}")

        st.subheader("Confusion matrix - ")
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots()
        color_matrix = np.array([
            [1, 0],
            [0, 1]
        ])
        ax2.imshow(color_matrix, cmap="RdYlGn")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, cm[i, j],
                         ha="center", va="center", color="black", fontsize=13, fontweight="bold")
        ax2.set_xticks([0,1])
        ax2.set_xticklabels(["NEGATIVE", "POSITIVE"], fontsize=11)
        ax2.set_yticks([0,1])
        ax2.set_yticklabels(["NEGATIVE", "POSITIVE"], fontsize=11)
        ax2.set_xlabel("PREDICTED LABEL", fontsize=14, labelpad=20)
        ax2.xaxis.set_label_position("top")
        ax2.xaxis.tick_top()
        ax2.set_ylabel("TRUE LABEL", fontsize=14, labelpad=10)

        st.pyplot(fig2)

        st.subheader("Dôležitosť atribútov")
        importance = clf.feature_importances_
        fig3, ax3 = plt.subplots()
        ax3.barh(features, importance)
        st.pyplot(fig3)

elif dataset_choice == "Titanic":
    st.title("Rozhodovací strom - Titanic")
    st.subheader("Prehľad datasetu Titanic")
    titanic_df = pd.read_csv("titanic.csv")
    titanic_df = titanic_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
    titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())
    titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
    titanic_df["Fare"] = titanic_df["Fare"].fillna(titanic_df["Fare"].median())


    titanic_df["Sex"] = titanic_df["Sex"].map({"male": 1, "female": 0})
    titanic_df["Embarked"] = titanic_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    st.write(titanic_df)

    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        st.metric("Celkový počet záznamov", len(titanic_df))
    with col_2:
        st.metric("Prežili", (titanic_df['Survived'] == 1).sum())
    with col_3:
        st.metric("Neprežili", (titanic_df['Survived'] == 0).sum())

    col_1, col_2, col_3, col_s4 = st.columns(4)
    with col_1:
        max_depth_t = st.slider("Hĺbka stromu", 1, 40, 3)
    with col_2:
        criterion_t = st.selectbox("Kritérium", ["gini", "entropy"])
    with col_3:
        min_samples_t = st.slider("Minimálne rozdelenie uzla", 2, 20, 2)
    with col_s4:
        test_size = st.slider("Testovacia sada (%)", 10, 40, 20)

    with st.expander("Kódovanie hodnôt"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Pohlavie (Sex)**")
            st.info("male = 1")
            st.info("female = 0")
        with col2:
            st.markdown("**Miesto nalodenia (Embarked)**")
            st.info("S (Southampton) = 0")
            st.info("C (Cherbourg) = 1")
            st.info("Q (Queenstown) = 2")
        with col3:
            st.markdown("**Trieda (Pclass)**")
            st.success("1. trieda = 1")
            st.warning("2. trieda = 2")
            st.error("3. trieda = 3")

    features_t = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = titanic_df[features_t].values
    y = titanic_df["Survived"].values

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X, y,
        test_size=test_size / 100,
        random_state=42,
        stratify=y
    )
    
    col_1, col_2 = st.columns(2)
    with col_1:
        st.metric("Tréningová sada: ", len(X_train_t))
    with col_2:
        st.metric("Testovacia sada: ", len(X_test_t))

    clf_t = DecisionTreeClassifier(
        criterion=criterion_t,
        max_depth=max_depth_t,
        min_samples_split=min_samples_t,
        random_state=42
    )
    clf_t.fit(X_train_t, y_train_t)

    _, counts = np.unique(y_train_t, return_counts=True)
    probs = counts / len(y_train_t)
    st.caption(
        f"Tréningová sada obsahuje {counts[0]} pasažierov, ktorí neprežili "
        f"a {counts[1]} ktorí prežili "
        f"z celkových {len(y_train_t)} záznamov."
    )

    y_pred_t = clf_t.predict(X_test_t)
    acc_t = accuracy_score(y_test_t, y_pred_t)

    st.success(f"Presnosť: {acc_t * 100:.2f}%  |  Hĺbka: {clf_t.get_depth()}  |  Uzlov: {clf_t.tree_.node_count}")

    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(clf_t, feature_names=features_t,
              class_names=["Neprežil", "Prežil"],
              filled=True, rounded=True, impurity=True, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test_t, y_pred_t)
    fig2, ax2 = plt.subplots()
    ax2.imshow([[1, 0], [0, 1]], cmap="RdYlGn")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i, j], ha="center", va="center",
                     color="black", fontsize=13, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["NEPREŽIL", "PREŽIL"], fontsize=11)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["NEPREŽIL", "PREŽIL"], fontsize=11)
    ax2.set_xlabel("PREDICTED LABEL", fontsize=13, labelpad=15)
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()
    ax2.set_ylabel("TRUE LABEL", fontsize=13, labelpad=10)
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Dôležitosť atribútov")
    fig3, ax3 = plt.subplots()
    ax3.barh(features_t, clf_t.feature_importances_)
    ax3.set_xlabel("Dôležitosť")
    st.pyplot(fig3)
    plt.close(fig3)


