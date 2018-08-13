# ANN Artifical Neural Network

Simple ANN with Kerls in Python:
  * Creating of ANN
  * k-Fold for parameter optimisation (sklearn)
  * Grid Search
  * Dropout for reduce of overfitting ANN

Dependency:
  * keras.models.Sequential
  * keras.layers.Dense
  * keras.wrappers.scikit_learn.KerasClassifier
  * sklearn.model_selection.GridSearchCV
  * pandas
  * sklearn.preprocessing.LabelEncoder, OneHotEncoder, StandardScaler
  * sklearn.model_selection.train_test_split




Inhaltsverzeichnis
==================

1. Gesamtkonzeption
    1.1. MISSION-Infrastruktur
    1.2. Routing und Events
        1.2.1. Das gesamte MISSION System: Überblick
        1.2.2. Routing Service: statische Sichtweise
            1.2.2.1. Definition und Aufgaben des Service	
            1.2.2.2. Architektur des Service
            1.2.2.3. Graphendatenbank Neo4j als Basis des Service
            1.2.2.4. Position des Routing Service im gesamten MISSION System
        1.2.3. Service Events: statische Sichtweise
            1.2.3.1. Definition und Aufgaben des Services
            1.2.3.2. Architektur des Services
            1.2.3.3. Event Store als Basis des Services
            1.2.3.4. Position des Event Service im gesamten MISSION System
        1.2.4. Funktionale und non-funktionale Anforderungen an MISSION System
            1.2.4.1. Anforderungen an den Routing Service
            1.2.4.1. Anforderungen an die Event Services
    1.3. Anwendungen
    1.4. Zusammenspiel der Komponenten
        1.4.1. Dienste werden in Infrastruktur registriert
        1.4.2. Anwendungen finden Dienste über Infrastruktur
        1.4.3. Anwendungen nutzen Dienste

Abkürzungsverzeichnis
=====================

    **API**		Anwendungsprogrammierschnittstelle
    **BRG**		Baltic Rail Gate GmbH
    **Bode**	Spedition Bode GmbH & CO. KG
    **CSV**		Comma-separated values
    **CQRS**	Command-Query-Resposibility-Segregation
    **DDD**		Domain-driven Design
    **ECL**		European Cargo Logistics GmbH
    **ETL**		Extraction, Transformation, Laden
    **GTFS**	General Transit Feed Specification
    **I/O**		Input / Output
    **LPG**		Label-Property Graphendatenbank
    **NoSQL**	Nicht nur SQL Datenbanktechnologien
    **RDBMS**	Relational Datenbankmanagementsystem
    **RDF**		Resource Description Framework
    **SQL**		Structured Query Language
