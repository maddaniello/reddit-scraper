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

class MultiAgentEvaluator:
    """Sistema multi-agente per valutazione approfondita dei post"""
    
    def __init__(self, client: OpenAI, business_info: Dict):
        self.client = client
        self.business_info = business_info
        self.logger = logging.getLogger(__name__)
    
    def agent_sensitivity_filter(self, post: Dict) -> Dict:
        """Agente 1: Filtra contenuti sensibili o inappropriati"""
        try:
            prompt = f"""
            Sei un esperto di reputazione aziendale e comunicazione sensibile.
            Analizza questo post Reddit per identificare se contiene elementi che renderebbero 
            inappropriato o rischioso un intervento aziendale.
            
            POST:
            Titolo: {post['title']}
            Contenuto: {post['selftext'][:1000]}
            
            CRITERI DI ESCLUSIONE IMMEDIATA:
            1. LUTTO/MORTE: Menzioni di morte, lutto, malattie terminali
            2. SALUTE MENTALE GRAVE: Depressione severa, pensieri suicidi, crisi psicologiche
            3. CONTROVERSIE: Discussioni politiche, religiose, etiche divisive
            4. LAMENTELE/SFOGHI: Rant contro aziende, servizi, professionisti
            5. SITUAZIONI PERSONALI DELICATE: Divorzi, violenze, abusi, problemi familiari gravi
            6. CONTENUTO ILLEGALE: Discussioni su attività illegali o al limite
            7. TONO AGGRESSIVO: Linguaggio offensivo, flame war, trolling
            8. PROBLEMI FINANZIARI GRAVI: Bancarotta, debiti insostenibili, disperazione economica
            9. DISCRIMINAZIONE: Qualsiasi forma di discriminazione o pregiudizio
            10. MINORI: Situazioni che coinvolgono minori in contesti delicati
            
            VALUTA ANCHE:
            - Il tono generale è costruttivo o distruttivo?
            - C'è spazio per un intervento professionale senza sembrare opportunisti?
            - Il contesto emotivo permetterebbe un intervento aziendale rispettoso?
            
            Rispondi SOLO in JSON:
            {{
                "is_sensitive": true/false,
                "sensitivity_score": 0-100 (100 = estremamente sensibile),
                "sensitivity_reasons": ["lista di motivi specifici"],
                "intervention_risk": "high/medium/low",
                "can_proceed": true/false
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
            except:
                # Fallback conservativo in caso di errore parsing
                result = {
                    "is_sensitive": True,
                    "sensitivity_score": 100,
                    "sensitivity_reasons": ["Errore valutazione"],
                    "intervention_risk": "high",
                    "can_proceed": False
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore agent_sensitivity_filter: {e}")
            return {
                "is_sensitive": True,
                "sensitivity_score": 100,
                "sensitivity_reasons": ["Errore sistema"],
                "intervention_risk": "high",
                "can_proceed": False
            }
    
    def agent_business_relevance(self, post: Dict) -> Dict:
        """Agente 2: Valuta la rilevanza per il business specifico"""
        try:
            prompt = f"""
            Sei un esperto di business development e marketing strategico.
            Valuta se questo post rappresenta un'opportunità reale per il business.
            
            BUSINESS CONTEXT:
            Brand: {self.business_info['brand_name']}
            Servizi: {self.business_info['services'][:500]}
            Value Proposition: {self.business_info.get('value_proposition', '')[:300]}
            
            POST:
            Titolo: {post['title']}
            Contenuto: {post['selftext'][:800]}
            Subreddit: r/{post['subreddit']}
            Commenti: {post['num_comments']}
            
            CRITERI DI VALUTAZIONE:
            1. ALLINEAMENTO SERVIZI: I nostri servizi risolvono direttamente il problema?
            2. TARGET AUDIENCE: L'autore è nel nostro target di clienti?
            3. STADIO DEL PROBLEMA: È nella fase giusta per il nostro intervento?
            4. COMPETENZA DIMOSTRATA: Possiamo dimostrare expertise specifica?
            5. VALORE AGGIUNTO: Possiamo offrire insights unici non già presenti nei commenti?
            6. TIMING: È il momento giusto per intervenire (non troppo tardi)?
            7. ROI POTENZIALE: Vale la pena investire tempo in questa conversazione?
            
            ESCLUDI SE:
            - Il problema è già risolto
            - Richiede competenze che non abbiamo
            - È una discussione teorica senza bisogno pratico
            - L'autore non sembra aperto a suggerimenti
            - È troppo generico o vago
            
            Rispondi SOLO in JSON:
            {{
                "business_alignment_score": 0-100,
                "is_target_audience": true/false,
                "problem_stage": "awareness/consideration/decision/solved",
                "value_we_can_add": "descrizione specifica",
                "competitive_advantage": "cosa ci distingue in questo caso",
                "should_engage": true/false
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
            except:
                result = {
                    "business_alignment_score": 0,
                    "is_target_audience": False,
                    "problem_stage": "unknown",
                    "value_we_can_add": "",
                    "competitive_advantage": "",
                    "should_engage": False
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore agent_business_relevance: {e}")
            return {
                "business_alignment_score": 0,
                "is_target_audience": False,
                "problem_stage": "unknown",
                "value_we_can_add": "",
                "competitive_advantage": "",
                "should_engage": False
            }
    
    def agent_engagement_strategy(self, post: Dict, sensitivity_result: Dict, business_result: Dict) -> Dict:
        """Agente 3: Definisce la strategia di engagement ottimale"""
        try:
            prompt = f"""
            Sei un esperto di community management e brand reputation.
            Basandoti sulle analisi precedenti, definisci la strategia di engagement.
            
            POST SUMMARY:
            Titolo: {post['title']}
            Score Reddit: {post['score']}
            Commenti: {post['num_comments']}
            È una domanda: {post.get('is_question', False)}
            
            ANALISI SENSIBILITÀ:
            Sensibilità: {sensitivity_result.get('sensitivity_score', 100)}/100
            Rischio: {sensitivity_result.get('intervention_risk', 'high')}
            
            ANALISI BUSINESS:
            Allineamento: {business_result.get('business_alignment_score', 0)}/100
            Target: {business_result.get('is_target_audience', False)}
            Fase: {business_result.get('problem_stage', 'unknown')}
            
            DEFINISCI:
            1. APPROCCIO: Come dovremmo rispondere (educativo, consultivo, supportivo)?
            2. TONO: Quale tono usare (professionale, amichevole, empatico)?
            3. CONTENUTO: Cosa includere/escludere nella risposta
            4. CTA: Quale call-to-action (se appropriata)
            5. RISCHI: Quali rischi monitorare
            6. PRIORITÀ: Quanto è prioritaria questa opportunità (1-10)
            
            Rispondi SOLO in JSON:
            {{
                "engagement_approach": "educational/consultative/supportive/none",
                "recommended_tone": "descrizione del tono",
                "key_points_to_address": ["punto 1", "punto 2"],
                "avoid_mentioning": ["cosa non dire"],
                "suggested_cta": "call to action se appropriata",
                "priority_score": 1-10,
                "estimated_impact": "high/medium/low",
                "final_recommendation": "ENGAGE/SKIP/MONITOR"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
            except:
                result = {
                    "engagement_approach": "none",
                    "recommended_tone": "",
                    "key_points_to_address": [],
                    "avoid_mentioning": [],
                    "suggested_cta": "",
                    "priority_score": 0,
                    "estimated_impact": "low",
                    "final_recommendation": "SKIP"
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore agent_engagement_strategy: {e}")
            return {
                "engagement_approach": "none",
                "recommended_tone": "",
                "key_points_to_address": [],
                "avoid_mentioning": [],
                "suggested_cta": "",
                "priority_score": 0,
                "estimated_impact": "low",
                "final_recommendation": "SKIP"
            }
    
    def evaluate_post_complete(self, post: Dict) -> Dict:
        """Valutazione completa multi-agente del post"""
        
        # Step 1: Filtro sensibilità
        sensitivity_result = self.agent_sensitivity_filter(post)
        
        # Se troppo sensibile, skip immediato
        if not sensitivity_result.get('can_proceed', False):
            return {
                'should_engage': False,
                'overall_score': 0,
                'sensitivity_score': sensitivity_result.get('sensitivity_score', 100),
                'business_alignment_score': 0,
                'priority_score': 0,
                'rejection_reason': f"Contenuto sensibile: {', '.join(sensitivity_result.get('sensitivity_reasons', []))}",
                'recommendation': 'SKIP',
                'details': {
                    'sensitivity': sensitivity_result
                }
            }
        
        # Step 2: Valutazione business
        business_result = self.agent_business_relevance(post)
        
        # Se non rilevante per il business, skip
        if not business_result.get('should_engage', False):
            return {
                'should_engage': False,
                'overall_score': business_result.get('business_alignment_score', 0),
                'sensitivity_score': sensitivity_result.get('sensitivity_score', 0),
                'business_alignment_score': business_result.get('business_alignment_score', 0),
                'priority_score': 0,
                'rejection_reason': 'Non allineato con il business o target',
                'recommendation': 'SKIP',
                'details': {
                    'sensitivity': sensitivity_result,
                    'business': business_result
                }
            }
        
        # Step 3: Strategia di engagement
        engagement_result = self.agent_engagement_strategy(post, sensitivity_result, business_result)
        
        # Calcolo score finale ponderato
        overall_score = (
            (100 - sensitivity_result.get('sensitivity_score', 0)) * 0.3 +  # Meno sensibile = meglio
            business_result.get('business_alignment_score', 0) * 0.4 +
            engagement_result.get('priority_score', 0) * 10 * 0.3
        )
        
        should_engage = (
            engagement_result.get('final_recommendation') == 'ENGAGE' and
            overall_score >= 60
        )
        
        return {
            'should_engage': should_engage,
            'overall_score': round(overall_score, 1),
            'sensitivity_score': sensitivity_result.get('sensitivity_score', 0),
            'business_alignment_score': business_result.get('business_alignment_score', 0),
            'priority_score': engagement_result.get('priority_score', 0),
            'recommendation': engagement_result.get('final_recommendation', 'SKIP'),
            'engagement_approach': engagement_result.get('engagement_approach', ''),
            'recommended_tone': engagement_result.get('recommended_tone', ''),
            'key_points': engagement_result.get('key_points_to_address', []),
            'avoid_mentioning': engagement_result.get('avoid_mentioning', []),
            'suggested_cta': engagement_result.get('suggested_cta', ''),
            'estimated_impact': engagement_result.get('estimated_impact', 'low'),
            'value_proposition': business_result.get('value_we_can_add', ''),
            'competitive_advantage': business_result.get('competitive_advantage', ''),
            'problem_stage': business_result.get('problem_stage', 'unknown'),
            'details': {
                'sensitivity': sensitivity_result,
                'business': business_result,
                'engagement': engagement_result
            }
        }


class RedditAIBusinessScraper:
    def __init__(self):
        """Inizializza il scraper Reddit con AI"""
        
        # Setup logging per Streamlit
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inizializza Reddit se le credenziali sono disponibili
        self.reddit = None
        self.client = None
        self.multi_agent = None
        
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
            # Inizializza il sistema multi-agente
            self.multi_agent = MultiAgentEvaluator(self.client, self.business_info)
            st.success("✅ OpenAI API e Multi-Agent System configurati")
        except Exception as e:
            st.error(f"❌ Errore OpenAI API: {e}")
            return False
        
        return True
    
    def _generate_value_proposition(self):
        """Genera una value proposition usando AI basata su chi siamo e servizi"""
        try:
            prompt = f"""
            Basandoti su queste informazioni aziendali, genera una dettagliata value proposition 
            che riassuma in che modo questa azienda può aiutare le persone:
            
            Chi siamo: {self.business_info['about_us']}
            
            Servizi: {self.business_info['services']}
            
            Genera una value proposition di massimo 5 righe che evidenzi:
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
                             limit_per_term: int = 50) -> List[Dict]:
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
                        'is_question': self._is_question_or_help_request(submission)
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
    
    def evaluate_with_multi_agent(self, posts: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Valuta i post con il sistema multi-agente"""
        
        # Aggiorna il multi-agent con le info business correnti
        self.multi_agent.business_info = self.business_info
        
        strategic_opportunities = []
        rejected_posts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, post in enumerate(posts):
            status_text.text(f"🤖 Analisi Multi-Agente post {i+1}/{len(posts)}")
            progress_bar.progress((i + 1) / len(posts))
            
            try:
                # Valutazione completa multi-agente
                evaluation = self.multi_agent.evaluate_post_complete(post)
                
                # Aggiungi i risultati al post
                post['multi_agent_evaluation'] = evaluation
                post['overall_score'] = evaluation['overall_score']
                post['sensitivity_score'] = evaluation['sensitivity_score']
                post['business_alignment_score'] = evaluation['business_alignment_score']
                post['priority_score'] = evaluation['priority_score']
                post['should_engage'] = evaluation['should_engage']
                post['recommendation'] = evaluation['recommendation']
                
                if evaluation['should_engage']:
                    # Aggiungi dettagli strategici
                    post['engagement_strategy'] = {
                        'approach': evaluation.get('engagement_approach', ''),
                        'tone': evaluation.get('recommended_tone', ''),
                        'key_points': evaluation.get('key_points', []),
                        'avoid': evaluation.get('avoid_mentioning', []),
                        'cta': evaluation.get('suggested_cta', ''),
                        'impact': evaluation.get('estimated_impact', ''),
                        'value_proposition': evaluation.get('value_proposition', ''),
                        'competitive_advantage': evaluation.get('competitive_advantage', ''),
                        'problem_stage': evaluation.get('problem_stage', '')
                    }
                    strategic_opportunities.append(post)
                else:
                    post['rejection_reason'] = evaluation.get('rejection_reason', 'Non strategico')
                    rejected_posts.append(post)
                
                # Rate limiting per OpenAI
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Errore valutazione multi-agente post {post['post_id']}: {e}")
                post['rejection_reason'] = f"Errore valutazione: {e}"
                rejected_posts.append(post)
        
        progress_bar.empty()
        status_text.empty()
        
        # Ordina le opportunità strategiche per score
        strategic_opportunities.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return strategic_opportunities, rejected_posts


def display_opportunity_card(opp: Dict, index: int):
    """Visualizza una card dettagliata per un'opportunità strategica"""
    
    with st.expander(f"🎯 #{index}. [Score: {opp['overall_score']:.1f}%] {opp['title'][:80]}..."):
        
        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Score Totale", f"{opp['overall_score']:.1f}%")
        with col2:
            st.metric("💼 Business Fit", f"{opp['business_alignment_score']:.0f}%")
        with col3:
            st.metric("⚠️ Sensibilità", f"{opp['sensitivity_score']:.0f}%", 
                     delta=f"Rischio {'Basso' if opp['sensitivity_score'] < 30 else 'Medio' if opp['sensitivity_score'] < 60 else 'Alto'}")
        with col4:
            st.metric("⭐ Priorità", f"{opp['priority_score']}/10")
        
        # Informazioni post
        st.write("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**📍 Subreddit:** r/{opp['subreddit']}")
            st.write(f"**👤 Autore:** {opp['author']}")
            st.write(f"**💬 Commenti:** {opp['num_comments']} | **⬆️ Score:** {opp['score']}")
            st.write(f"**🔍 Trovato con:** {opp['search_term']}")
            
            if opp['selftext']:
                st.write("**📝 Contenuto:**")
                st.write(opp['selftext'][:500] + ("..." if len(opp['selftext']) > 500 else ""))
        
        with col2:
            # Strategia di engagement
            strategy = opp.get('engagement_strategy', {})
            
            st.write("**🎯 Strategia Engagement:**")
            st.info(f"**Approccio:** {strategy.get('approach', 'N/A').title()}")
            st.info(f"**Tono:** {strategy.get('tone', 'N/A')}")
            st.info(f"**Impatto:** {strategy.get('impact', 'N/A').title()}")
            
            if strategy.get('problem_stage'):
                stage_emoji = {
                    'awareness': '👁️',
                    'consideration': '🤔',
                    'decision': '✅',
                    'solved': '🏆'
                }.get(strategy.get('problem_stage'), '❓')
                st.write(f"**{stage_emoji} Fase:** {strategy.get('problem_stage', '').title()}")
        
        # Punti chiave e cosa evitare
        col1, col2 = st.columns(2)
        
        with col1:
            if strategy.get('key_points'):
                st.write("**✅ Punti da Affrontare:**")
                for point in strategy.get('key_points', []):
                    st.write(f"• {point}")
        
        with col2:
            if strategy.get('avoid'):
                st.write("**❌ Da Evitare:**")
                for avoid in strategy.get('avoid', []):
                    st.write(f"• {avoid}")
        
        # Value proposition e vantaggio competitivo
        if strategy.get('value_proposition'):
            st.write("**💡 Valore che Possiamo Offrire:**")
            st.success(strategy.get('value_proposition'))
        
        if strategy.get('competitive_advantage'):
            st.write("**🏆 Nostro Vantaggio Competitivo:**")
            st.info(strategy.get('competitive_advantage'))
        
        # Call to Action suggerita
        if strategy.get('cta'):
            st.write("**📣 Call to Action Suggerita:**")
            st.warning(strategy.get('cta'))
        
        # Link al post
        st.markdown(f"[🔗 Vai al post Reddit]({opp['reddit_url']})")


def main():
    st.title("🤖 Reddit AI Business Scraper - Multi-Agent Edition")
    st.markdown("### Sistema avanzato di identificazione opportunità strategiche")
    
    # Sidebar per configurazione
    with st.sidebar:
        st.header("🔧 Configurazione API")
        
        reddit_client_id = st.text_input("Reddit Client ID", type="password")
        reddit_client_secret = st.text_input("Reddit Client Secret", type="password")
        reddit_user_agent = st.text_input("Reddit User Agent", value="BusinessScraper/2.0")
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
        about_us = st.text_area("📝 Chi Siamo", height=150, 
                                help="Descrivi brevemente la tua azienda, missione e valori")
    
    with col2:
        services = st.text_area("📋 Servizi Offerti", height=150,
                               help="Elenca i servizi che offri e come aiuti i clienti")
    
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
            placeholder="mutuo, prestito personale, cessione del quinto",
            help="Inserisci i termini di ricerca principali"
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
            help="Espande automaticamente le keyword con sinonimi e variazioni"
        )
        
        min_overall_score = st.slider(
            "🎯 Score Minimo Richiesto",
            min_value=50,
            max_value=80,
            value=60,
            step=5,
            help="Solo opportunità con score complessivo >= a questo valore"
        )
    
    # Opzioni avanzate
    with st.expander("⚙️ Opzioni Avanzate"):
        col1, col2 = st.columns(2)
        with col1:
            max_sensitivity = st.slider(
                "⚠️ Sensibilità Massima Accettabile",
                min_value=20,
                max_value=60,
                value=40,
                help="Post con sensibilità superiore saranno esclusi"
            )
        with col2:
            min_priority = st.slider(
                "⭐ Priorità Minima",
                min_value=3,
                max_value=8,
                value=5,
                help="Solo opportunità con priorità >= a questo valore"
            )
    
    # Avvia ricerca
    if st.button("🚀 Avvia Analisi Multi-Agente", type="primary") and base_topics:
        
        if not (brand_name and about_us and services):
            st.error("❌ Completa prima la configurazione business!")
            return
        
        st.header("📊 Risultati Analisi")
        
        # 1. Espansione termini (se richiesta)
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
        st.subheader("🌐 Fase 1: Ricerca su Reddit")
        with st.spinner("Ricerca post in corso..."):
            all_posts = scraper.search_reddit_globally(unique_search_terms, time_filter, limit_per_term=30)
        
        if not all_posts:
            st.warning("❌ Nessun post trovato!")
            return
        
        st.success(f"✅ Trovati {len(all_posts)} post italiani pertinenti")
        
        # 3. Analisi Multi-Agente
        st.subheader("🤖 Fase 2: Analisi Multi-Agente")
        with st.spinner(f"Analisi strategica di {len(all_posts)} post..."):
            strategic_opportunities, rejected_posts = scraper.evaluate_with_multi_agent(all_posts)
        
        # Applica filtri aggiuntivi
        filtered_opportunities = [
            opp for opp in strategic_opportunities 
            if opp['overall_score'] >= min_overall_score 
            and opp['sensitivity_score'] <= max_sensitivity
            and opp['priority_score'] >= min_priority
        ]
        
        # 4. Risultati
        st.header("🎯 Risultati Finali")
        
        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Post Analizzati", len(all_posts))
        
        with col2:
            st.metric("✅ Opportunità Strategiche", len(filtered_opportunities),
                     delta=f"-{len(all_posts) - len(filtered_opportunities)} filtrati")
        
        with col3:
            if filtered_opportunities:
                avg_score = sum(o['overall_score'] for o in filtered_opportunities) / len(filtered_opportunities)
                st.metric("📈 Score Medio", f"{avg_score:.1f}%")
            else:
                st.metric("📈 Score Medio", "N/A")
        
        with col4:
            high_priority = len([o for o in filtered_opportunities if o['priority_score'] >= 7])
            st.metric("🔥 Alta Priorità", high_priority)
        
        # 5. Analisi dettagliata opportunità
        if filtered_opportunities:
            st.subheader(f"🏆 {len(filtered_opportunities)} Opportunità Strategiche Identificate")
            
            # Filtri visualizzazione
            col1, col2, col3 = st.columns(3)
            with col1:
                show_approach = st.selectbox(
                    "Filtra per Approccio",
                    ["Tutti"] + list(set(o.get('engagement_strategy', {}).get('approach', '') 
                                       for o in filtered_opportunities if o.get('engagement_strategy', {}).get('approach')))
                )
            with col2:
                show_impact = st.selectbox(
                    "Filtra per Impatto",
                    ["Tutti", "high", "medium", "low"]
                )
            with col3:
                sort_by = st.selectbox(
                    "Ordina per",
                    ["Score Totale", "Business Alignment", "Priorità", "Sensibilità"]
                )
            
            # Applica filtri e ordinamento
            displayed_opportunities = filtered_opportunities.copy()
            
            if show_approach != "Tutti":
                displayed_opportunities = [o for o in displayed_opportunities 
                                         if o.get('engagement_strategy', {}).get('approach') == show_approach]
            
            if show_impact != "Tutti":
                displayed_opportunities = [o for o in displayed_opportunities 
                                         if o.get('engagement_strategy', {}).get('impact') == show_impact]
            
            # Ordinamento
            sort_keys = {
                "Score Totale": lambda x: x['overall_score'],
                "Business Alignment": lambda x: x['business_alignment_score'],
                "Priorità": lambda x: x['priority_score'],
                "Sensibilità": lambda x: -x['sensitivity_score']  # Inverso per sensibilità
            }
            displayed_opportunities.sort(key=sort_keys[sort_by], reverse=True)
            
            # Visualizza opportunità
            for i, opp in enumerate(displayed_opportunities[:20], 1):
                display_opportunity_card(opp, i)
        
        else:
            st.warning("❌ Nessuna opportunità strategica trovata con i criteri specificati")
            
            # Mostra alcuni post rifiutati per context
            if rejected_posts:
                with st.expander(f"📊 Analisi {len(rejected_posts)} post esclusi"):
                    rejection_reasons = {}
                    for post in rejected_posts[:10]:
                        reason = post.get('rejection_reason', 'Non specificato')
                        if reason not in rejection_reasons:
                            rejection_reasons[reason] = 0
                        rejection_reasons[reason] += 1
                    
                    st.write("**Motivi di esclusione:**")
                    for reason, count in rejection_reasons.items():
                        st.write(f"• {reason}: {count} post")
        
        # 6. Export risultati
        if filtered_opportunities:
            st.subheader("💾 Export Risultati")
            
            # Prepara dati per export
            export_data = []
            for opp in filtered_opportunities:
                strategy = opp.get('engagement_strategy', {})
                export_data.append({
                    'score_totale': opp['overall_score'],
                    'business_alignment': opp['business_alignment_score'],
                    'sensitivita': opp['sensitivity_score'],
                    'priorita': opp['priority_score'],
                    'subreddit': opp['subreddit'],
                    'titolo': opp['title'],
                    'contenuto': opp['selftext'][:500] if opp['selftext'] else '',
                    'autore': opp['author'],
                    'url_reddit': opp['reddit_url'],
                    'num_commenti': opp['num_comments'],
                    'approccio': strategy.get('approach', ''),
                    'tono_consigliato': strategy.get('tone', ''),
                    'punti_chiave': '|'.join(strategy.get('key_points', [])),
                    'da_evitare': '|'.join(strategy.get('avoid', [])),
                    'cta_suggerita': strategy.get('cta', ''),
                    'valore_offerto': strategy.get('value_proposition', ''),
                    'vantaggio_competitivo': strategy.get('competitive_advantage', ''),
                    'fase_problema': strategy.get('problem_stage', ''),
                    'impatto_stimato': strategy.get('impact', '')
                })
            
            df = pd.DataFrame(export_data)
            
            # CSV download
            csv = df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📊 Scarica CSV Completo",
                    data=csv,
                    file_name=f"reddit_opportunities_strategic_{timestamp}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON download con tutti i dettagli
                json_data = {
                    'timestamp': timestamp,
                    'business': brand_name,
                    'keywords': base_topics,
                    'total_posts_analyzed': len(all_posts),
                    'strategic_opportunities': len(filtered_opportunities),
                    'filters_applied': {
                        'min_overall_score': min_overall_score,
                        'max_sensitivity': max_sensitivity,
                        'min_priority': min_priority
                    },
                    'opportunities': [
                        {
                            **opp,
                            'created_utc': opp['created_utc'].isoformat() if isinstance(opp['created_utc'], datetime) else str(opp['created_utc'])
                        } 
                        for opp in filtered_opportunities
                    ]
                }
                
                json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📄 Scarica JSON Dettagliato",
                    data=json_str,
                    file_name=f"reddit_opportunities_detailed_{timestamp}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
