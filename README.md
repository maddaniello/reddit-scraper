# Reddit AI Business Scraper - Streamlit App

Un'applicazione web che utilizza l'intelligenza artificiale per identificare opportunità di business su Reddit, analizzando discussioni e domande pertinenti al tuo settore.

## Funzionalità

- **Ricerca Globale**: Cerca su tutto Reddit (non solo subreddit specifici)
- **Espansione Automatica**: Espande automaticamente i termini di ricerca con sinonimi e variazioni
- **Filtro Italiano**: Filtra automaticamente solo contenuti in italiano
- **Analisi AI**: Utilizza OpenAI per identificare opportunità business concrete
- **Report Interattivi**: Genera report dettagliati con score di rilevanza
- **Export Dati**: Esporta risultati in CSV e JSON

## Requisiti

### API Keys Necessarie

1. **Reddit API**:
   - Vai su [Reddit Apps](https://www.reddit.com/prefs/apps)
   - Crea una nuova app di tipo "script"
   - Ottieni `client_id` e `client_secret`

2. **OpenAI API**:
   - Vai su [OpenAI API Keys](https://platform.openai.com/api-keys)
   - Genera una nuova API key

### Installazione Locale

```bash
# Clona il repository
git clone https://github.com/tuousername/reddit-ai-scraper.git
cd reddit-ai-scraper

# Installa le dipendenze
pip install -r requirements.txt

# Avvia l'app
streamlit run app.py
```

## Deploy su Streamlit Cloud

1. **Fork questo repository** su GitHub
2. Vai su [Streamlit Cloud](https://streamlit.io/cloud)
3. Connetti il tuo account GitHub
4. Seleziona il repository forkato
5. Imposta il file principale: `app.py`
6. Deploy!

## Come Usare

1. **Configura le API**: Inserisci le tue credenziali nella sidebar
2. **Contesto Business**: Compila le informazioni sulla tua azienda
3. **Keywords**: Inserisci le parole chiave separate da virgola
4. **Avvia Ricerca**: Clicca il pulsante per iniziare l'analisi
5. **Esamina Risultati**: Visualizza le opportunità trovate
6. **Export**: Scarica i risultati in CSV o JSON

## Sicurezza

- **Mai inserire API keys nel codice**
- Le credenziali vengono inserite tramite l'interfaccia web
- Streamlit Cloud supporta secrets per le variabili d'ambiente

## Configurazione Secrets (Streamlit Cloud)

Se vuoi evitare di inserire le API keys ogni volta:

1. Vai nelle impostazioni dell'app su Streamlit Cloud
2. Aggiungi i secrets:

```toml
REDDIT_CLIENT_ID = "tuo_client_id"
REDDIT_CLIENT_SECRET = "tuo_client_secret"
REDDIT_USER_AGENT = "BusinessScraper/1.0"
OPENAI_API_KEY = "tua_openai_key"
```

## Limitazioni

- **Rate Limiting**: Rispetta i limiti API di Reddit e OpenAI
- **Contenuto Italiano**: Filtra automaticamente solo contenuti in italiano
- **Costi OpenAI**: Ogni analisi AI ha un piccolo costo (circa $0.001 per post)

## Contributi

Le pull request sono benvenute! Per modifiche importanti, apri prima un issue.

## Licenza

MIT License - vedi il file LICENSE per i dettagli
