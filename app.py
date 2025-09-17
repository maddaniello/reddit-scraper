import streamlit as st
import praw
import pandas as pd
import os
import json
import time
from datetime import datetime, timedelta
import logging
from collections import Counter
import re
from openai import OpenAI
from typing import List, Dict, Set, Tuple
import io
import base64
import zipfile

# Configurazione della pagina
st.set_page_config(
    page_title="Reddit AI Business Scraper",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RedditAIBusinessScraper:
    def __init__(self):
        """Inizializza il scraper Reddit con AI"""
        
        # Setup logging per Streamlit
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inizializza Reddit se le credenziali sono disponibili
        self.reddit = None
        self.client = None
        
        # Informazioni business (verranno popolate dall'utente)
        self.business_info = {
            'brand_name': '',
            'about_us': '',
            'services': '',
            'value_proposition': ''
        }
        
        # Espansione termini di ricerca italiani
        self.search_expansions = {
            'mutuo': {
                'sinonimi': ['mutui', 'finanziamento', 'finanziamenti', 'prestito', 'prestiti', 
                            'credito', 'ipoteca', 'ipoteche', 'mutuo casa', 'mutuo prima casa'],
                'contesti': ['banca', 'banche', 'tasso', 'tassi', 'spread', 'rata', 'rate', 
                            'surroga', 'surroghe', 'broker', 'consulente', 'taeg', 'tan', 
                            'piano ammortamento', 'rogito', 'notaio', 'perizia'],
                'problemi': ['rifiutato', 'negato', 'problemi', 'difficoltà', 'aiuto', 'consiglio',
                            'non riesco', 'bocciato', 'respinto', 'documenti', 'garanzie']
            },
            'casa': {
                'sinonimi': ['case', 'abitazione', 'abitazioni', 'immobile', 'immobili', 
                           'appartamento', 'appartamenti', 'villa', 'ville', 'bilocale', 'trilocale'],
                'contesti': ['acquisto', 'acquistare', 'comprare', 'vendita', 'vendere', 
                           'affitto', 'affittare', 'agenzia', 'agente immobiliare', 'ristrutturazione',
                           'ristrutturare', 'classe energetica', 'catasto'],
                'problemi': ['cerco', 'cercando', 'trovare', 'budget', 'prezzi', 'mercato',
                           'valutazione', 'perizia', 'compromesso', 'proposta']
            },
            'investimento': {
                'sinonimi': ['investimenti', 'investire', 'portafoglio', 'rendita', 'rendimento'],
                'contesti': ['immobiliare', 'mattone', 'locazione', 'airbnb', 'affitti brevi',
                           'rivalutazione', 'plusvalenza', 'tasse', 'imu', 'cedolare secca'],
                'problemi': ['conviene', 'meglio', 'consiglio', 'strategia', 'rischio', 'rendimento']
            }
        }
        
        # Pattern per identificare domande e richieste di aiuto
        self.help_patterns = [
            r'\?',  # Domande
            r'(aiuto|aiutatemi|help)',
            r'(consiglio|consigli|suggerimento)',
            r'(qualcuno sa|qualcuno ha|qualcuno può)',
            r'(come posso|come fare|come si fa)',
            r'(cosa ne pensate|che ne pensate|opinioni)',
            r'(conviene|meglio|preferibile)',
            r'(problema|problemi|difficoltà)',
            r'(non so|non riesco|non capisco)',
            r'(cercando|cerco|in cerca di)',
            r'(esperienza|esperienze)',
            r'(dubbio|dubbi|incerto)'
        ]
    
    def initialize_apis(self, reddit_client_id, reddit_client_secret, reddit_user_agent, openai_api_key):
        """Inizializza le API con le credenziali fornite"""
        try:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
            st.success("✅ Reddit API configurato")
        except Exception as e:
            st.error(f"❌ Errore Reddit API: {e}")
            return False
        
        try:
            self.client = OpenAI(api_key=openai_api_key)
            st.success("✅ OpenAI API configurato")
        except Exception as e:
            st.error(f"❌ Errore OpenAI API: {e}")
            return False
        
        return True
    
    def _generate_value_proposition(self):
        """Genera una value proposition usando AI basata su chi siamo e servizi"""
        try:
            prompt = f"""
            Basandoti su queste informazioni aziendali, genera una breve value proposition 
            che riassuma in che modo questa azienda può aiutare le persone:
            
            Chi siamo: {self.business_info['about_us']}
            
            Servizi: {self.business_info['services']}
            
            Genera una value proposition di massimo 3 righe che evidenzi:
            1. Quali problemi risolve
            2. A chi si rivolge
            3. Qual è il valore unico che offre
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            self.business_info['value_proposition'] = response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Errore generazione value proposition: {e}")
            self.business_info['value_proposition'] = "Aiutiamo i clienti con soluzioni professionali"
    
    def _generate_services_optimization(self):
        """Ottimizza la descrizione dei servizi usando AI"""
        try:
            prompt = f"""
            Ottimizza questa descrizione dei servizi per renderla più efficace nel contesto business e marketing:
            
            Servizi attuali: {self.business_info['services']}
            
            Contesto aziendale: {self.business_info['about_us']}
            
            Genera una versione ottimizzata che:
            1. Sia più chiara e diretta
            2. Evidenzi i benefici per il cliente
            3. Utilizzi un linguaggio persuasivo ma professionale
            4. Sia strutturata in modo leggibile
            
            Mantieni la stessa lunghezza approssimativa del testo originale.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Errore ottimizzazione servizi: {e}")
            return self.business_info['services']
    
    def expand_search_terms(self, base_topic: str) -> List[str]:
        """Espande i termini di ricerca con sinonimi e contesti"""
        expanded_terms = [base_topic]
        base_lower = base_topic.lower()
        
        # Cerca espansioni predefinite
        for key, expansions in self.search_expansions.items():
            if key in base_lower or base_lower in key:
                expanded_terms.extend(expansions['sinonimi'])
                # Aggiungi combinazioni con contesti
                for context in expansions['contesti'][:5]:  # Limita per non esagerare
                    expanded_terms.append(f"{base_topic} {context}")
                # Aggiungi problemi comuni
                for problem in expansions['problemi'][:3]:
                    expanded_terms.append(f"{problem} {base_topic}")
                break
        
        # Genera anche variazioni con AI
        try:
            ai_expansions = self._generate_search_variations_ai(base_topic)
            expanded_terms.extend(ai_expansions)
        except:
            pass
        
        # Rimuovi duplicati mantenendo l'ordine
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return unique_terms[:15]  # Massimo 15 termini
    
    def _generate_search_variations_ai(self, topic: str) -> List[str]:
        """Usa AI per generare variazioni di ricerca"""
        try:
            prompt = f"""
            Genera 5 variazioni in italiano del termine di ricerca "{topic}" per Reddit.
            Include sinonimi, forme plurali, e frasi correlate che le persone userebbero.
            Rispondi solo con i termini separati da virgola, senza numerazione.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5
            )
            
            variations = response.choices[0].message.content.strip().split(',')
            return [v.strip() for v in variations if v.strip()][:5]
            
        except Exception as e:
            self.logger.error(f"Errore generazione variazioni AI: {e}")
            return []
    
    def search_reddit_globally(self, search_terms: List[str], time_filter: str = 'month', 
                             limit_per_term: int = 50, use_ai_expansion: bool = True) -> List[Dict]:
        """Ricerca globale su tutto Reddit per ogni termine"""
        all_posts = []
        processed_ids = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, term in enumerate(search_terms):
            status_text.text(f"🔍 Ricerca globale per: '{term}' ({i+1}/{len(search_terms)})")
            progress_bar.progress((i + 1) / len(search_terms))
            
            try:
                # Ricerca su tutto Reddit
                search_results = self.reddit.subreddit('all').search(
                    term, 
                    time_filter=time_filter,
                    limit=limit_per_term,
                    sort='relevance'
                )
                
                for submission in search_results:
                    if submission.id in processed_ids:
                        continue
                    
                    processed_ids.add(submission.id)
                    
                    # Verifica che sia contenuto italiano
                    if not self._is_italian_content(submission.title + " " + submission.selftext):
                        continue
                    
                    post_data = {
                        'post_id': submission.id,
                        'subreddit': submission.subreddit.display_name,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'author': str(submission.author) if submission.author else '[deleted]',
                        'url': submission.url,
                        'reddit_url': f"https://reddit.com{submission.permalink}",
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'created_utc': datetime.fromtimestamp(submission.created_utc),
                        'search_term': term,
                        'is_question': self._is_question_or_help_request(submission),
                        'has_opportunity': False,  # Verrà valutato dopo con AI
                        'ai_relevance_score': 0,
                        'ai_opportunity_reason': ''
                    }
                    
                    all_posts.append(post_data)
                
                time.sleep(1)  # Rate limiting tra ricerche
                
            except Exception as e:
                st.error(f"Errore ricerca per '{term}': {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return all_posts
    
    def _is_italian_content(self, text: str) -> bool:
        """Verifica veloce se il contenuto è italiano"""
        if not text or len(text) < 20:
            return False
        
        italian_indicators = [
            'sono', 'è', 'ho', 'hai', 'abbiamo', 'hanno',
            'dove', 'quando', 'come', 'perché', 'cosa', 'chi',
            'molto', 'poco', 'tutto', 'niente', 'qualcuno',
            'grazie', 'prego', 'scusa', 'ciao', 'salve',
            'della', 'dello', 'delle', 'degli', 'nella', 'sulla'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for ind in italian_indicators if ind in text_lower)
        
        return matches >= 3
    
    def _is_question_or_help_request(self, submission) -> bool:
        """Identifica se è una domanda o richiesta di aiuto"""
        text = f"{submission.title} {submission.selftext}".lower()
        
        for pattern in self.help_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def evaluate_business_relevance_ai(self, posts: List[Dict]) -> List[Dict]:
        """Valuta con AI quali post sono opportunità per il business"""
        
        evaluated_posts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, post in enumerate(posts):
            status_text.text(f"🤖 Valutazione AI post {i+1}/{len(posts)}")
            progress_bar.progress((i + 1) / len(posts))
            
            try:
                # Prepara il contesto per l'AI
                evaluation = self._evaluate_single_post_ai(post)
                
                post['has_opportunity'] = evaluation['has_opportunity']
                post['ai_relevance_score'] = evaluation['relevance_score']
                post['ai_opportunity_reason'] = evaluation['reason']
                post['suggested_response_angle'] = evaluation.get('response_angle', '')
                
                evaluated_posts.append(post)
                
                # Rate limiting per OpenAI
                time.sleep(0.5)
                
            except Exception as e:
                st.error(f"Errore valutazione post {post['post_id']}: {e}")
                post['has_opportunity'] = False
                post['ai_relevance_score'] = 0
                evaluated_posts.append(post)
        
        progress_bar.empty()
        status_text.empty()
        
        # Filtra solo le opportunità
        opportunities = [p for p in evaluated_posts if p['has_opportunity']]
        
        return opportunities
    
    def _evaluate_single_post_ai(self, post: Dict) -> Dict:
        """Valuta un singolo post con AI - versione più stringente"""
        try:
            # Estrai primi commenti se disponibili
            comments_context = ""
            if post['num_comments'] > 0:
                try:
                    submission = self.reddit.submission(id=post['post_id'])
                    submission.comments.replace_more(limit=0)
                    top_comments = submission.comments[:3]
                    comments_context = "\n".join([c.body[:200] for c in top_comments])
                except:
                    pass
            
            prompt = f"""
            Analizza con CRITERI MOLTO STRINGENTI questa conversazione Reddit per identificare se rappresenta un'opportunità CONCRETA di fornire valore aggiunto senza apparire promozionali.
            
            CONTESTO BUSINESS:
            Brand: {self.business_info['brand_name']}
            Chi siamo: {self.business_info['about_us'][:500]}
            Servizi: {self.business_info['services'][:500]}
            
            POST REDDIT:
            Titolo: {post['title']}
            Testo: {post['selftext'][:800]}
            Commenti: {post['num_comments']} commenti
            Top commenti: {comments_context}
            
            CRITERI STRINGENTI (DEVE SODDISFARE TUTTI):
            1. DOMANDA SPECIFICA: C'è una domanda diretta e specifica che richiede expertise professionale?
            2. COMPETENZA RILEVANTE: La nostra competenza è direttamente applicabile al problema?
            3. VALORE AUTENTICO: Possiamo fornire un consiglio genuino e non promozionale?
            4. CONTESTO APPROPRIATO: Il nostro intervento sarà ben accolto dalla community?
            5. CONVERSAZIONE ATTIVA: Ci sono già altre risposte o la discussione è attiva?
            6. AUTORITÀ NATURALE: Il nostro intervento può posizionarci naturalmente come esperti?
            
            ESCLUDI IMMEDIATAMENTE SE:
            - Post troppo generici o vaghi
            - Già troppe risposte di esperti
            - Tono ostile o controverso
            - Contenuto non professionale
            - Domanda già risolta nei commenti
            - Richiede competenze che non abbiamo
            
            PUNTEGGIO:
            - 80-100: Opportunità eccellente, intervento altamente raccomandato
            - 60-79: Buona opportunità, intervento raccomandato
            - 50-59: Opportunità marginale, valutare attentamente
            - <50: Sconsigliato, filtrare
            
            Rispondi in formato JSON:
            {{
                "has_opportunity": true/false,
                "relevance_score": 0-100,
                "reason": "analisi dettagliata dei criteri",
                "response_angle": "approccio specifico raccomandato",
                "intervention_type": "educational/consultative/supportive",
                "urgency": "high/medium/low"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2  # Più deterministico per criteri stringenti
            )
            
            # Parse della risposta
            result_text = response.choices[0].message.content.strip()
            
            # Prova a parsare come JSON
            try:
                result = json.loads(result_text)
            except:
                # Fallback se non è JSON valido
                result = {
                    "has_opportunity": "true" in result_text.lower() and "high" in result_text.lower(),
                    "relevance_score": 30,  # Score basso di default per fallback
                    "reason": result_text[:150],
                    "response_angle": "",
                    "intervention_type": "educational",
                    "urgency": "low"
                }
            
            # Applica filtro stringente: solo score >= 50
            if result.get("relevance_score", 0) < 50:
                result["has_opportunity"] = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore valutazione AI: {e}")
            return {
                "has_opportunity": False,
                "relevance_score": 0,
                "reason": "Errore valutazione",
                "response_angle": "",
                "intervention_type": "none",
                "urgency": "low"
            }

def create_download_files(opportunities: List[Dict], base_topics: List[str], business_name: str):
    """Crea i file per il download"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_string = "_".join([t.replace(" ", "-") for t in base_topics])
    
    # 1. CSV delle opportunità
    opp_data = []
    for post in opportunities:
        opp_data.append({
            'relevance_score': post['ai_relevance_score'],
            'subreddit': post['subreddit'],
            'title': post['title'],
            'post_text': post['selftext'],
            'author': post['author'],
            'reddit_url': post['reddit_url'],
            'score': post['score'],
            'comments': post['num_comments'],
            'created': post['created_utc'],
            'is_question': post['is_question'],
            'opportunity_reason': post['ai_opportunity_reason'],
            'suggested_approach': post.get('suggested_response_angle', ''),
            'search_term': post['search_term'],
            'intervention_type': post.get('intervention_type', ''),
            'urgency': post.get('urgency', '')
        })
    
    # Ordina per relevance score
    opp_data.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # CSV
    csv_buffer = io.StringIO()
    pd.DataFrame(opp_data).to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_content = csv_buffer.getvalue()
    
    # JSON
    json_data = {
        'keywords': base_topics,
        'business': business_name,
        'timestamp': timestamp,
        'total_opportunities': len(opportunities),
        'opportunities': opportunities
    }
    json_content = json.dumps(json_data, indent=2, ensure_ascii=False, default=str)
    
    return csv_content, json_content, f"{topic_string}_opportunities_{timestamp}"

def analyze_opportunities_trends(opportunities: List[Dict]) -> Dict:
    """Analizza i trend e pattern nelle opportunità trovate"""
    if not opportunities:
        return {}
    
    # Analisi argomenti più frequenti
    all_titles = " ".join([opp['title'].lower() for opp in opportunities])
    all_content = " ".join([opp['selftext'].lower()[:200] for opp in opportunities if opp['selftext']])
    
    # Parole chiave più frequenti (filtrate con stopwords estese)
    import re
    words = re.findall(r'\b[a-záàéèíìóòúù]{4,}\b', all_titles + " " + all_content)
    
    # Stopwords estese per italiano
    stop_words = {
        # Articoli, preposizioni, congiunzioni
        'sono', 'della', 'delle', 'dello', 'degli', 'nella', 'nelle', 'nello', 'negli',
        'dalla', 'dalle', 'dallo', 'dagli', 'alla', 'alle', 'allo', 'agli', 'sulla',
        'sulle', 'sullo', 'sugli', 'per', 'con', 'una', 'uno', 'che', 'più', 'come',
        'dove', 'quando', 'cosa', 'qualcuno', 'molto', 'poco', 'tutto', 'anche',
        'ancora', 'sempre', 'mai', 'già', 'prima', 'dopo', 'oggi', 'ieri', 'domani',
        'questa', 'questo', 'questi', 'queste', 'quello', 'quella', 'quelli', 'quelle',
        'stesso', 'stessa', 'stessi', 'stesse', 'altro', 'altra', 'altri', 'altre',
        
        # Verbi comuni
        'vorrei', 'voglio', 'vuole', 'volevo', 'devo', 'deve', 'dovrei', 'posso',
        'può', 'potrei', 'facendo', 'fare', 'fatto', 'faccio', 'fanno', 'essere',
        'stato', 'stata', 'stati', 'state', 'avere', 'aveva', 'avevo', 'avrò',
        'dire', 'detto', 'dice', 'dicono', 'andare', 'vado', 'viene', 'venire',
        'mettere', 'messo', 'prendere', 'preso', 'vedere', 'visto', 'sentire',
        'sentito', 'sapere', 'sapevo', 'uscire', 'entrare', 'rimanere', 'stare',
        'dare', 'dato', 'portare', 'portato', 'trovare', 'trovato',
        
        # Avverbi e aggettivi generici
        'bene', 'male', 'meglio', 'peggio', 'abbastanza', 'troppo', 'tanto',
        'poco', 'parecchio', 'davvero', 'veramente', 'proprio', 'solo', 'soltanto',
        'piuttosto', 'invece', 'infatti', 'inoltre', 'quindi', 'però', 'però',
        'comunque', 'tuttavia', 'mentre', 'durante', 'dentro', 'fuori', 'sopra',
        'sotto', 'vicino', 'lontano', 'grande', 'piccolo', 'nuovo', 'vecchio',
        'giovane', 'primo', 'ultimo', 'buono', 'cattivo', 'bello', 'brutto',
        
        # Pronomi e particelle
        'loro', 'nostro', 'nostra', 'nostri', 'nostre', 'vostro', 'vostra',
        'vostri', 'vostre', 'ogni', 'alcuni', 'alcune', 'nessuno', 'nessuna',
        'ciascuno', 'ciascuna', 'chiunque', 'ovunque', 'dovunque', 'qualsiasi',
        'qualche', 'niente', 'nulla', 'qualcosa', 'qualcosa',
        
        # Parole di contesto Reddit
        'post', 'thread', 'commento', 'commenti', 'utente', 'utenti', 'forum',
        'discussione', 'opinione', 'opinioni', 'reddit', 'subreddit',
        
        # Numeri e date
        'anno', 'anni', 'mese', 'mesi', 'settimana', 'settimane', 'giorno',
        'giorni', 'ora', 'ore', 'minuto', 'minuti', 'tempo', 'volta', 'volte'
    }
    
    # Filtra parole e mantieni solo sostantivi rilevanti
    filtered_words = []
    for word in words:
        if (word not in stop_words and 
            len(word) > 4 and  # Almeno 5 caratteri
            not word.isdigit() and  # Non numeri
            not any(char.isdigit() for char in word)):  # Non contiene numeri
            filtered_words.append(word)
    
    word_freq = Counter(filtered_words)
    
    # Subreddit più attivi
    subreddit_freq = Counter([opp['subreddit'] for opp in opportunities])
    
    # Analisi sentiment base
    question_keywords = ['aiuto', 'problema', 'difficoltà', 'non riesco', 'consiglio', 'dubbio']
    positive_keywords = ['grazie', 'ottimo', 'perfetto', 'bene', 'soddisfatto']
    negative_keywords = ['male', 'sbagliato', 'pessimo', 'deluso', 'problema', 'difficoltà']
    
    questions_count = sum(1 for opp in opportunities if any(kw in opp['title'].lower() + " " + opp['selftext'].lower() for kw in question_keywords))
    
    # Tipi di intervento
    intervention_types = Counter([opp.get('intervention_type', 'unknown') for opp in opportunities])
    urgency_levels = Counter([opp.get('urgency', 'unknown') for opp in opportunities])
    
    # Score distribution
    score_ranges = {
        '90-100%': len([o for o in opportunities if o['ai_relevance_score'] >= 90]),
        '80-89%': len([o for o in opportunities if 80 <= o['ai_relevance_score'] < 90]),
        '70-79%': len([o for o in opportunities if 70 <= o['ai_relevance_score'] < 80]),
        '60-69%': len([o for o in opportunities if 60 <= o['ai_relevance_score'] < 70]),
        '50-59%': len([o for o in opportunities if 50 <= o['ai_relevance_score'] < 60])
    }
    
    # Timing analysis
    hours = [opp['created_utc'].hour for opp in opportunities]
    hour_freq = Counter(hours)
    
    return {
        'top_keywords': word_freq.most_common(10),
        'top_subreddits': subreddit_freq.most_common(5),
        'questions_percentage': round((questions_count / len(opportunities)) * 100, 1),
        'intervention_types': dict(intervention_types),
        'urgency_levels': dict(urgency_levels),
        'score_distribution': score_ranges,
        'peak_hours': hour_freq.most_common(3),
        'avg_score': round(sum(o['ai_relevance_score'] for o in opportunities) / len(opportunities), 1),
        'avg_comments': round(sum(o['num_comments'] for o in opportunities) / len(opportunities), 1)
    }

def main():
    st.title("🤖 Reddit AI Business Scraper")
    st.markdown("### Ricerca intelligente con filtro opportunità business")
    
    # Sidebar per configurazione
    with st.sidebar:
        st.header("🔧 Configurazione API")
        
        reddit_client_id = st.text_input("Reddit Client ID", type="password")
        reddit_client_secret = st.text_input("Reddit Client Secret", type="password")
        reddit_user_agent = st.text_input("Reddit User Agent", value="BusinessScraper/1.0")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
        apis_configured = all([reddit_client_id, reddit_client_secret, reddit_user_agent, openai_api_key])
        
        if apis_configured:
            st.success("✅ Credenziali inserite")
        else:
            st.warning("⚠️ Inserisci tutte le credenziali")
    
    # Inizializza scraper
    scraper = RedditAIBusinessScraper()
    
    if not apis_configured:
        st.info("👈 Inserisci le credenziali API nella sidebar per iniziare")
        return
    
    # Configura API
    if not scraper.initialize_apis(reddit_client_id, reddit_client_secret, reddit_user_agent, openai_api_key):
        st.error("❌ Errore configurazione API")
        return
    
    # Configurazione business
    st.header("🏢 Configurazione Contesto Business")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand_name = st.text_input("🏷️ Nome del Brand/Azienda")
        about_us = st.text_area("📝 Chi Siamo", height=150)
    
    with col2:
        services = st.text_area("📋 Servizi", height=150)
        if services and st.button("🔧 Ottimizza Servizi con AI"):
            scraper.business_info['services'] = services
            with st.spinner("Ottimizzazione servizi..."):
                optimized_services = scraper._generate_services_optimization()
                st.success("✅ Servizi ottimizzati!")
                st.text_area("📋 Servizi Ottimizzati", value=optimized_services, height=150, key="optimized_services")
                if st.button("✅ Usa Versione Ottimizzata"):
                    services = optimized_services
                    st.rerun()
    
    if brand_name and about_us and services:
        scraper.business_info = {
            'brand_name': brand_name,
            'about_us': about_us,
            'services': services,
            'value_proposition': ''
        }
        
        # Genera value proposition
        if st.button("🎯 Genera Value Proposition con AI"):
            with st.spinner("Generazione value proposition..."):
                scraper._generate_value_proposition()
                st.success("✅ Value proposition generata!")
                st.info(f"💡 **Value Proposition:** {scraper.business_info['value_proposition']}")
    
    # Configurazione ricerca
    st.header("🎯 Configurazione Ricerca")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        keywords_input = st.text_input(
            "🔍 Keywords (separate da virgola)",
            placeholder="mutuo, prestito personale, cessione del quinto"
        )
        
        base_topics = [t.strip() for t in keywords_input.split(',') if t.strip()]
    
    with col2:
        time_filter = st.selectbox(
            "⏰ Periodo di ricerca",
            options=['week', 'month', 'year'],
            format_func=lambda x: {
                'week': '📅 Ultima settimana',
                'month': '📅 Ultimo mese (consigliato)',
                'year': '📅 Ultimo anno'
            }[x],
            index=1
        )
    
    with col3:
        use_ai_expansion = st.checkbox(
            "🤖 Espansione AI Keywords",
            value=True,
            help="Espande automaticamente le keyword con sinonimi e variazioni usando AI"
        )
        
        relevance_threshold = st.slider(
            "🎯 Soglia Rilevanza Minima",
            min_value=50,
            max_value=90,
            value=60,
            step=5,
            help="Solo opportunità con score >= a questo valore"
        )
    
    # Avvia ricerca
    if st.button("🚀 Avvia Ricerca Intelligente", type="primary") and base_topics:
        
        if not (brand_name and about_us and services):
            st.error("❌ Completa prima la configurazione business!")
            return
        
        st.header("📊 Risultati Ricerca")
        
        # 1. Espansione termini (opzionale)
        if use_ai_expansion:
            with st.expander("🔍 Espansione termini di ricerca"):
                all_search_terms = []
                for topic in base_topics:
                    expanded = scraper.expand_search_terms(topic)
                    all_search_terms.extend(expanded)
                    st.write(f"**{topic}:** {', '.join(expanded[:8])}")
                
                # Rimuovi duplicati
                seen = set()
                unique_search_terms = []
                for term in all_search_terms:
                    if term.lower() not in seen:
                        seen.add(term.lower())
                        unique_search_terms.append(term)
                
                st.info(f"📊 Totale termini unici: {len(unique_search_terms)}")
        else:
            unique_search_terms = base_topics
            st.info(f"🔍 Ricerca diretta per: {', '.join(base_topics)}")
        
        # 2. Ricerca Reddit
        st.subheader("🌐 Ricerca globale su Reddit")
        with st.spinner("Ricerca in corso..."):
            all_posts = scraper.search_reddit_globally(unique_search_terms, time_filter, limit_per_term=30, use_ai_expansion=use_ai_expansion)
        
        if not all_posts:
            st.warning("❌ Nessun post trovato!")
            return
        
        st.success(f"📊 Trovati {len(all_posts)} post italiani pertinenti")
        
        # 3. Valutazione AI
        st.subheader("🤖 Analisi AI delle opportunità")
        with st.spinner(f"Analisi di {len(all_posts)} post in corso..."):
            all_evaluated = scraper.evaluate_business_relevance_ai(all_posts)
            # Applica soglia di rilevanza
            opportunities = [opp for opp in all_evaluated if opp['ai_relevance_score'] >= relevance_threshold]
        
        if not opportunities:
            st.warning(f"❌ Nessuna opportunità trovata con score >= {relevance_threshold}%")
            if all_evaluated:
                st.info(f"💡 {len(all_evaluated)} post trovati ma filtrati per bassa rilevanza. Prova ad abbassare la soglia.")
            return
        
        # 4. Risultati e Statistiche
        st.success(f"✅ TROVATE {len(opportunities)} OPPORTUNITÀ QUALIFICATE!")
        
        # Analisi trend
        trends = analyze_opportunities_trends(opportunities)
        
        # Statistiche principali
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Opportunità Filtrate", len(opportunities))
        
        with col2:
            high_relevance = len([o for o in opportunities if o['ai_relevance_score'] >= 80])
            st.metric("🔥 Score Elevato (80%+)", high_relevance)
        
        with col3:
            questions = len([o for o in opportunities if o['is_question']])
            st.metric("❓ Domande Dirette", questions)
        
        with col4:
            st.metric("📈 Score Medio", f"{trends['avg_score']}%")
        
        # 5. Recap Analitico
        st.header("📊 Analisi Approfondita")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏷️ Argomenti Più Frequenti")
            if trends['top_keywords']:
                for word, count in trends['top_keywords']:
                    percentage = round((count / len(opportunities)) * 100, 1)
                    st.write(f"• **{word.title()}**: {count} volte ({percentage}%)")
            
            st.subheader("📍 Subreddit Più Attivi")
            if trends['top_subreddits']:
                for subreddit, count in trends['top_subreddits']:
                    percentage = round((count / len(opportunities)) * 100, 1)
                    st.write(f"• **r/{subreddit}**: {count} opportunità ({percentage}%)")
        
        with col2:
            st.subheader("🎯 Distribuzione Score")
            for range_name, count in trends['score_distribution'].items():
                if count > 0:
                    percentage = round((count / len(opportunities)) * 100, 1)
                    st.write(f"• **{range_name}**: {count} opportunità ({percentage}%)")
            
            st.subheader("🕐 Orari di Picco")
            if trends['peak_hours']:
                for hour, count in trends['peak_hours']:
                    st.write(f"• **{hour:02d}:00**: {count} post")
        
        # Insight aggiuntivi
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("❓ % Domande", f"{trends['questions_percentage']}%")
        
        with col2:
            st.metric("💬 Commenti Medi", f"{trends['avg_comments']}")
        
        with col3:
            if trends['intervention_types']:
                most_common_type = max(trends['intervention_types'].items(), key=lambda x: x[1])
                st.metric("🎯 Tipo Principale", most_common_type[0].title())
        
        # 6. Insight Strategici
        st.subheader("🧠 Insight Strategici")
        
        insights = []
        
        if trends['questions_percentage'] > 70:
            insights.append("🎯 **Alto tasso di domande**: Ottima opportunità per posizionarsi come esperti")
        
        if trends['avg_score'] > 75:
            insights.append("⭐ **Score elevato**: Le opportunità identificate sono di alta qualità")
        
        if len(trends['top_subreddits']) <= 2:
            insights.append("📍 **Concentrazione geografica**: Focus su pochi subreddit specifici")
        
        if any(hour < 9 or hour > 18 for hour, _ in trends['peak_hours']):
            insights.append("🕐 **Attività fuori orario**: Molte discussioni avvengono la sera/notte")
        
        if trends.get('urgency_levels', {}).get('high', 0) > len(opportunities) * 0.3:
            insights.append("🚨 **Urgenza elevata**: Molte opportunità richiedono intervento rapido")
        
        for insight in insights:
            st.info(insight)
        
        # 7. Top opportunità dettagliate
        st.subheader("🏆 Top Opportunità")
        
        for i, opp in enumerate(sorted(opportunities, key=lambda x: x['ai_relevance_score'], reverse=True)[:10], 1):
            with st.expander(f"{i}. [Score: {opp['ai_relevance_score']}%] {opp['title'][:80]}..."):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**📍 Subreddit:** r/{opp['subreddit']}")
                    st.write(f"**👤 Autore:** {opp['author']}")
                    st.write(f"**🔍 Trovato con:** {opp['search_term']}")
                    
                    # Nuovi campi
                    if opp.get('intervention_type'):
                        st.write(f"**🎯 Tipo Intervento:** {opp['intervention_type'].title()}")
                    if opp.get('urgency'):
                        urgency_emoji = {'high': '🚨', 'medium': '⚠️', 'low': '📝'}.get(opp['urgency'], '📝')
                        st.write(f"**⏰ Urgenza:** {urgency_emoji} {opp['urgency'].title()}")
                    
                    if opp['selftext']:
                        st.write("**📝 Contenuto:**")
                        st.write(opp['selftext'][:300] + ("..." if len(opp['selftext']) > 300 else ""))
                
                with col2:
                    st.metric("⬆️ Score", opp['score'])
                    st.metric("💬 Commenti", opp['num_comments'])
                    st.write(f"**❓ Domanda:** {'Sì' if opp['is_question'] else 'No'}")
                
                st.write("**💡 Opportunità:**")
                st.info(opp['ai_opportunity_reason'])
                
                if opp.get('suggested_response_angle'):
                    st.write("**📝 Approccio suggerito:**")
                    st.success(opp['suggested_response_angle'])
                
                st.markdown(f"[🔗 Vai al post Reddit]({opp['reddit_url']})")
        
        # 8. Download files con session state
        st.subheader("💾 Download Risultati")
        
        # Salva i risultati in session_state per evitare che scompaiano
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {
                'opportunities': opportunities,
                'trends': trends,
                'base_topics': base_topics,
                'brand_name': brand_name
            }
        
        csv_content, json_content, filename_base = create_download_files(
            st.session_state.analysis_results['opportunities'], 
            st.session_state.analysis_results['base_topics'], 
            st.session_state.analysis_results['brand_name']
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📊 Scarica CSV",
                data=csv_content,
                file_name=f"{filename_base}.csv",
                mime="text/csv",
                key="download_csv"
            )
        
        with col2:
            st.download_button(
                label="📄 Scarica JSON",
                data=json_content,
                file_name=f"{filename_base}.json",
                mime="application/json",
                key="download_json"
            )
        
        with col3:
            if st.button("🔄 Nuova Analisi", key="reset_analysis"):
                # Pulisce i risultati salvati per permettere una nuova analisi
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
                st.rerun()

if __name__ == "__main__":
    main()
