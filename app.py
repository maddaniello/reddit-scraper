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
from typing import List, Dict, Set, Tuple, Optional
import io
import base64
import zipfile
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import pickle
import hashlib

# Configurazione della pagina
st.set_page_config(
    page_title="Reddit AI Business Scraper - Multi-Language",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurazione lingue supportate
SUPPORTED_LANGUAGES = {
    'it': {
        'name': 'Italiano 🇮🇹',
        'indicators': ['sono', 'è', 'ho', 'hai', 'della', 'nella', 'cosa', 'come', 'quando', 'perché'],
        'help_patterns': [r'\?', r'aiuto', r'consiglio', r'qualcuno', r'come posso', r'problema'],
        'subreddits': ['italy', 'italyinformatica', 'commercialisti', 'economia', 'finanza']
    },
    'en': {
        'name': 'English 🇬🇧',
        'indicators': ['the', 'is', 'are', 'have', 'has', 'what', 'when', 'where', 'why', 'how'],
        'help_patterns': [r'\?', r'help', r'advice', r'anyone', r'how can', r'problem', r'issue'],
        'subreddits': ['business', 'entrepreneur', 'smallbusiness', 'finance', 'personalfinance']
    },
    'es': {
        'name': 'Español 🇪🇸',
        'indicators': ['es', 'está', 'son', 'tiene', 'qué', 'cuando', 'donde', 'por qué', 'cómo'],
        'help_patterns': [r'\?', r'ayuda', r'consejo', r'alguien', r'cómo puedo', r'problema'],
        'subreddits': ['spain', 'espanol', 'negocios', 'finanzas', 'emprendedores']
    },
    'fr': {
        'name': 'Français 🇫🇷',
        'indicators': ['le', 'la', 'est', 'sont', 'avoir', 'que', 'quand', 'où', 'pourquoi', 'comment'],
        'help_patterns': [r'\?', r'aide', r'conseil', r'quelqu\'un', r'comment', r'problème'],
        'subreddits': ['france', 'vosfinances', 'conseiljuridique', 'entreprise']
    },
    'de': {
        'name': 'Deutsch 🇩🇪',
        'indicators': ['der', 'die', 'das', 'ist', 'sind', 'haben', 'was', 'wann', 'wo', 'warum'],
        'help_patterns': [r'\?', r'hilfe', r'rat', r'jemand', r'wie kann', r'problem'],
        'subreddits': ['de', 'finanzen', 'selbststaendig', 'wirtschaft']
    }
}

class SessionLogger:
    """Gestisce il logging delle sessioni in un formato strutturato"""
    
    def __init__(self):
        self.log_file = "reddit_scraper_sessions.json"
        self.current_session = {
            'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'language': None,
            'keywords': [],
            'posts_found': 0,
            'opportunities_identified': 0,
            'errors': [],
            'filters_applied': {},
            'execution_time_seconds': 0
        }
    
    def update_session(self, **kwargs):
        """Aggiorna i dati della sessione corrente"""
        for key, value in kwargs.items():
            if key in self.current_session:
                self.current_session[key] = value
    
    def log_error(self, error_msg: str):
        """Registra un errore nella sessione"""
        self.current_session['errors'].append({
            'time': datetime.now().isoformat(),
            'message': str(error_msg)
        })
    
    def save_session(self):
        """Salva la sessione corrente nel file di log"""
        try:
            # Calcola tempo di esecuzione
            if self.current_session['start_time']:
                start = datetime.fromisoformat(self.current_session['start_time'])
                self.current_session['execution_time_seconds'] = (datetime.now() - start).total_seconds()
            
            self.current_session['end_time'] = datetime.now().isoformat()
            
            # Carica log esistente o crea nuovo
            try:
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
            
            # Aggiungi sessione corrente
            logs.append(self.current_session)
            
            # Salva
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Errore salvataggio log: {e}")
            return False
    
    def export_to_excel(self) -> bytes:
        """Esporta i log in formato Excel"""
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Converti in DataFrame
            df_data = []
            for log in logs:
                df_data.append({
                    'Session ID': log.get('session_id', ''),
                    'Start Time': log.get('start_time', ''),
                    'End Time': log.get('end_time', ''),
                    'Language': log.get('language', ''),
                    'Keywords': ', '.join(log.get('keywords', [])),
                    'Posts Found': log.get('posts_found', 0),
                    'Opportunities': log.get('opportunities_identified', 0),
                    'Execution Time (s)': log.get('execution_time_seconds', 0),
                    'Errors Count': len(log.get('errors', []))
                })
            
            df = pd.DataFrame(df_data)
            
            # Crea Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Sessions Log', index=False)
            
            return output.getvalue()
            
        except Exception as e:
            st.error(f"Errore export Excel log: {e}")
            return None

class CheckpointManager:
    """Gestisce checkpoint e recupero dello stato"""
    
    def __init__(self):
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = None
    
    def create_checkpoint(self, data: Dict, checkpoint_name: str = None):
        """Crea un checkpoint dello stato corrente"""
        try:
            if not checkpoint_name:
                checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
            
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
            
            return True
        except Exception as e:
            st.error(f"Errore creazione checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """Carica un checkpoint salvato"""
        try:
            filepath = os.path.join(self.checkpoint_dir, checkpoint_file)
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Errore caricamento checkpoint: {e}")
            return None
    
    def list_checkpoints(self) -> List[str]:
        """Elenca tutti i checkpoint disponibili"""
        try:
            return [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl')]
        except:
            return []
    
    def delete_checkpoint(self, checkpoint_file: str):
        """Elimina un checkpoint"""
        try:
            filepath = os.path.join(self.checkpoint_dir, checkpoint_file)
            os.remove(filepath)
            return True
        except:
            return False

class MultiAgentEvaluator:
    """Sistema multi-agente per valutazione approfondita dei post"""
    
    def __init__(self, client: OpenAI, business_info: Dict, language: str = 'it'):
        self.client = client
        self.business_info = business_info
        self.language = language
        self.logger = logging.getLogger(__name__)
        self.timeout_seconds = 30  # Timeout per chiamata AI
    
    def agent_sensitivity_filter(self, post: Dict) -> Dict:
        """Agente 1: Filtra contenuti sensibili o inappropriati"""
        try:
            # Adatta il prompt alla lingua
            lang_instruction = f"Analyze this post in {SUPPORTED_LANGUAGES[self.language]['name']} language."
            
            prompt = f"""
            You are an expert in corporate reputation and sensitive communication.
            {lang_instruction}
            Analyze this Reddit post to identify if it contains elements that would make 
            a business intervention inappropriate or risky.
            
            POST:
            Title: {post['title']}
            Content: {post['selftext'][:1000]}
            
            IMMEDIATE EXCLUSION CRITERIA:
            1. DEATH/MOURNING: Mentions of death, mourning, terminal illnesses
            2. SEVERE MENTAL HEALTH: Severe depression, suicidal thoughts, psychological crises
            3. CONTROVERSIES: Divisive political, religious, ethical discussions
            4. COMPLAINTS/RANTS: Rants against companies, services, professionals
            5. DELICATE PERSONAL SITUATIONS: Divorces, violence, abuse, serious family problems
            6. ILLEGAL CONTENT: Discussions about illegal or borderline activities
            7. AGGRESSIVE TONE: Offensive language, flame wars, trolling
            8. SERIOUS FINANCIAL PROBLEMS: Bankruptcy, unsustainable debts, economic desperation
            9. DISCRIMINATION: Any form of discrimination or prejudice
            10. MINORS: Situations involving minors in sensitive contexts
            
            ALSO EVALUATE:
            - Is the general tone constructive or destructive?
            - Is there room for professional intervention without appearing opportunistic?
            - Would the emotional context allow for respectful business intervention?
            
            Reply ONLY in JSON:
            {{
                "is_sensitive": true/false,
                "sensitivity_score": 0-100 (100 = extremely sensitive),
                "sensitivity_reasons": ["list of specific reasons"],
                "intervention_risk": "high/medium/low",
                "can_proceed": true/false
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
                timeout=self.timeout_seconds
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
            except:
                result = {
                    "is_sensitive": True,
                    "sensitivity_score": 100,
                    "sensitivity_reasons": ["Parse error"],
                    "intervention_risk": "high",
                    "can_proceed": False
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error agent_sensitivity_filter: {e}")
            return {
                "is_sensitive": True,
                "sensitivity_score": 100,
                "sensitivity_reasons": ["System error"],
                "intervention_risk": "high",
                "can_proceed": False
            }
    
    def agent_business_relevance(self, post: Dict) -> Dict:
        """Agente 2: Valuta la rilevanza per il business specifico"""
        try:
            lang_instruction = f"Analyze in {SUPPORTED_LANGUAGES[self.language]['name']} language."
            
            prompt = f"""
            You are an expert in business development and strategic marketing.
            {lang_instruction}
            Evaluate if this post represents a real opportunity for the business.
            
            BUSINESS CONTEXT:
            Brand: {self.business_info['brand_name']}
            Services: {self.business_info['services'][:500]}
            Value Proposition: {self.business_info.get('value_proposition', '')[:300]}
            
            POST:
            Title: {post['title']}
            Content: {post['selftext'][:800]}
            Subreddit: r/{post['subreddit']}
            Comments: {post['num_comments']}
            
            EVALUATION CRITERIA:
            1. SERVICE ALIGNMENT: Do our services directly solve the problem?
            2. TARGET AUDIENCE: Is the author in our target customer base?
            3. PROBLEM STAGE: Is it at the right stage for our intervention?
            4. DEMONSTRATED EXPERTISE: Can we demonstrate specific expertise?
            5. ADDED VALUE: Can we offer unique insights not already in comments?
            6. TIMING: Is it the right time to intervene (not too late)?
            7. POTENTIAL ROI: Is it worth investing time in this conversation?
            
            EXCLUDE IF:
            - The problem is already solved
            - Requires skills we don't have
            - It's a theoretical discussion without practical need
            - The author doesn't seem open to suggestions
            - It's too generic or vague
            
            Reply ONLY in JSON:
            {{
                "business_alignment_score": 0-100,
                "is_target_audience": true/false,
                "problem_stage": "awareness/consideration/decision/solved",
                "value_we_can_add": "specific description",
                "competitive_advantage": "what distinguishes us in this case",
                "should_engage": true/false
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.2,
                timeout=self.timeout_seconds
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
            self.logger.error(f"Error agent_business_relevance: {e}")
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
            lang_instruction = f"Provide strategy for {SUPPORTED_LANGUAGES[self.language]['name']} market."
            
            prompt = f"""
            You are an expert in community management and brand reputation.
            {lang_instruction}
            Based on previous analyses, define the engagement strategy.
            
            POST SUMMARY:
            Title: {post['title']}
            Reddit Score: {post['score']}
            Comments: {post['num_comments']}
            Is question: {post.get('is_question', False)}
            
            SENSITIVITY ANALYSIS:
            Sensitivity: {sensitivity_result.get('sensitivity_score', 100)}/100
            Risk: {sensitivity_result.get('intervention_risk', 'high')}
            
            BUSINESS ANALYSIS:
            Alignment: {business_result.get('business_alignment_score', 0)}/100
            Target: {business_result.get('is_target_audience', False)}
            Stage: {business_result.get('problem_stage', 'unknown')}
            
            DEFINE:
            1. APPROACH: How should we respond (educational, consultative, supportive)?
            2. TONE: What tone to use (professional, friendly, empathetic)?
            3. CONTENT: What to include/exclude in the response
            4. CTA: What call-to-action (if appropriate)
            5. RISKS: What risks to monitor
            6. PRIORITY: How priority is this opportunity (1-10)
            
            Reply ONLY in JSON:
            {{
                "engagement_approach": "educational/consultative/supportive/none",
                "recommended_tone": "tone description",
                "key_points_to_address": ["point 1", "point 2"],
                "avoid_mentioning": ["what not to say"],
                "suggested_cta": "call to action if appropriate",
                "priority_score": 1-10,
                "estimated_impact": "high/medium/low",
                "final_recommendation": "ENGAGE/SKIP/MONITOR"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
                timeout=self.timeout_seconds
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
            self.logger.error(f"Error agent_engagement_strategy: {e}")
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
        """Valutazione completa multi-agente del post con timeout"""
        
        start_time = time.time()
        
        # Step 1: Filtro sensibilità
        sensitivity_result = self.agent_sensitivity_filter(post)
        
        # Check timeout
        if time.time() - start_time > self.timeout_seconds * 3:
            return {
                'should_engage': False,
                'overall_score': 0,
                'rejection_reason': 'Timeout during evaluation'
            }
        
        # Se troppo sensibile, skip immediato
        if not sensitivity_result.get('can_proceed', False):
            return {
                'should_engage': False,
                'overall_score': 0,
                'sensitivity_score': sensitivity_result.get('sensitivity_score', 100),
                'business_alignment_score': 0,
                'priority_score': 0,
                'rejection_reason': f"Sensitive content: {', '.join(sensitivity_result.get('sensitivity_reasons', []))}",
                'recommendation': 'SKIP',
                'details': {
                    'sensitivity': sensitivity_result
                }
            }
        
        # Step 2: Valutazione business
        business_result = self.agent_business_relevance(post)
        
        # Check timeout
        if time.time() - start_time > self.timeout_seconds * 3:
            return {
                'should_engage': False,
                'overall_score': 0,
                'rejection_reason': 'Timeout during evaluation'
            }
        
        # Se non rilevante per il business, skip
        if not business_result.get('should_engage', False):
            return {
                'should_engage': False,
                'overall_score': business_result.get('business_alignment_score', 0),
                'sensitivity_score': sensitivity_result.get('sensitivity_score', 0),
                'business_alignment_score': business_result.get('business_alignment_score', 0),
                'priority_score': 0,
                'rejection_reason': 'Not aligned with business or target',
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
            (100 - sensitivity_result.get('sensitivity_score', 0)) * 0.3 +
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
    def __init__(self, language: str = 'it'):
        """Inizializza il scraper Reddit con AI"""
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Language configuration
        self.language = language
        self.language_config = SUPPORTED_LANGUAGES[language]
        
        # Initialize components
        self.reddit = None
        self.client = None
        self.multi_agent = None
        self.session_logger = SessionLogger()
        self.checkpoint_manager = CheckpointManager()
        
        # Business info
        self.business_info = {
            'brand_name': '',
            'about_us': '',
            'services': '',
            'value_proposition': ''
        }
        
        # Timeout settings
        self.search_timeout = 60  # secondi per ricerca
        self.evaluation_timeout = 120  # secondi per valutazione
    
    def initialize_apis(self, reddit_client_id, reddit_client_secret, reddit_user_agent, openai_api_key):
        """Inizializza le API con le credenziali fornite"""
        try:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
            st.success("✅ Reddit API configured")
        except Exception as e:
            st.error(f"❌ Reddit API Error: {e}")
            self.session_logger.log_error(f"Reddit API initialization failed: {e}")
            return False
        
        try:
            self.client = OpenAI(api_key=openai_api_key)
            self.multi_agent = MultiAgentEvaluator(self.client, self.business_info, self.language)
            st.success("✅ OpenAI API and Multi-Agent System configured")
        except Exception as e:
            st.error(f"❌ OpenAI API Error: {e}")
            self.session_logger.log_error(f"OpenAI API initialization failed: {e}")
            return False
        
        return True
    
    def _is_target_language_content(self, text: str) -> bool:
        """Verifica se il contenuto è nella lingua target"""
        if not text or len(text) < 20:
            return False
        
        indicators = self.language_config['indicators']
        text_lower = text.lower()
        matches = sum(1 for ind in indicators if ind in text_lower)
        
        return matches >= 3
    
    def _is_question_or_help_request(self, submission) -> bool:
        """Identifica se è una domanda o richiesta di aiuto"""
        text = f"{submission.title} {submission.selftext}".lower()
        
        for pattern in self.language_config['help_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def search_reddit_with_checkpoint(self, search_terms: List[str], time_filter: str = 'month', 
                                    limit_per_term: int = 30, subreddits: List[str] = None) -> List[Dict]:
        """Ricerca su Reddit con supporto checkpoint"""
        all_posts = []
        processed_ids = set()
        
        # Usa subreddit suggeriti per la lingua se non specificati
        if not subreddits:
            subreddits = ['all']  # Ricerca globale di default
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Crea checkpoint iniziale
        checkpoint_data = {
            'search_terms': search_terms,
            'processed_terms': [],
            'all_posts': [],
            'processed_ids': set()
        }
        
        for i, term in enumerate(search_terms):
            status_text.text(f"🔍 Searching for: '{term}' ({i+1}/{len(search_terms)})")
            progress_bar.progress((i + 1) / len(search_terms))
            
            try:
                # Timeout per la ricerca
                search_start = time.time()
                
                for subreddit_name in subreddits:
                    if time.time() - search_start > self.search_timeout:
                        st.warning(f"⚠️ Timeout searching '{term}' in r/{subreddit_name}")
                        break
                    
                    try:
                        if subreddit_name == 'all':
                            search_results = self.reddit.subreddit('all').search(
                                term, 
                                time_filter=time_filter,
                                limit=limit_per_term,
                                sort='relevance'
                            )
                        else:
                            search_results = self.reddit.subreddit(subreddit_name).search(
                                term,
                                time_filter=time_filter,
                                limit=limit_per_term,
                                sort='relevance'
                            )
                        
                        for submission in search_results:
                            if submission.id in processed_ids:
                                continue
                            
                            processed_ids.add(submission.id)
                            
                            # Verifica lingua
                            if not self._is_target_language_content(submission.title + " " + submission.selftext):
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
                    
                    except Exception as e:
                        st.error(f"Error searching r/{subreddit_name}: {e}")
                        self.session_logger.log_error(f"Search error in r/{subreddit_name}: {e}")
                        continue
                
                # Salva checkpoint dopo ogni termine
                checkpoint_data['processed_terms'].append(term)
                checkpoint_data['all_posts'] = all_posts
                checkpoint_data['processed_ids'] = processed_ids
                self.checkpoint_manager.create_checkpoint(checkpoint_data)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                st.error(f"Error searching for '{term}': {e}")
                self.session_logger.log_error(f"Search error for '{term}': {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return all_posts
    
    def evaluate_with_multi_agent_checkpoint(self, posts: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Valuta i post con sistema multi-agente e checkpoint"""
        
        self.multi_agent.business_info = self.business_info
        
        strategic_opportunities = []
        rejected_posts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Checkpoint per valutazione
        checkpoint_data = {
            'evaluated_posts': 0,
            'strategic_opportunities': [],
            'rejected_posts': []
        }
        
        batch_size = 10  # Processa in batch per checkpoint frequenti
        
        for batch_start in range(0, len(posts), batch_size):
            batch_end = min(batch_start + batch_size, len(posts))
            batch = posts[batch_start:batch_end]
            
            for i, post in enumerate(batch, batch_start):
                status_text.text(f"🤖 Multi-Agent Analysis {i+1}/{len(posts)}")
                progress_bar.progress((i + 1) / len(posts))
                
                try:
                    # Timeout per valutazione
                    eval_start = time.time()
                    
                    if time.time() - eval_start > self.evaluation_timeout:
                        st.warning(f"⚠️ Timeout evaluating post {post['post_id']}")
                        post['rejection_reason'] = 'Evaluation timeout'
                        rejected_posts.append(post)
                        continue
                    
                    evaluation = self.multi_agent.evaluate_post_complete(post)
                    
                    # Aggiungi risultati al post
                    post['multi_agent_evaluation'] = evaluation
                    post['overall_score'] = evaluation['overall_score']
                    post['sensitivity_score'] = evaluation['sensitivity_score']
                    post['business_alignment_score'] = evaluation['business_alignment_score']
                    post['priority_score'] = evaluation['priority_score']
                    post['should_engage'] = evaluation['should_engage']
                    post['recommendation'] = evaluation['recommendation']
                    
                    if evaluation['should_engage']:
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
                        post['rejection_reason'] = evaluation.get('rejection_reason', 'Not strategic')
                        rejected_posts.append(post)
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating post {post['post_id']}: {e}")
                    self.session_logger.log_error(f"Evaluation error: {e}")
                    post['rejection_reason'] = f"Evaluation error: {e}"
                    rejected_posts.append(post)
            
            # Checkpoint dopo ogni batch
            checkpoint_data['evaluated_posts'] = batch_end
            checkpoint_data['strategic_opportunities'] = strategic_opportunities
            checkpoint_data['rejected_posts'] = rejected_posts
            self.checkpoint_manager.create_checkpoint(checkpoint_data, f"evaluation_{batch_end}")
        
        progress_bar.empty()
        status_text.empty()
        
        strategic_opportunities.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return strategic_opportunities, rejected_posts


def export_to_excel(opportunities: List[Dict], filename: str = None) -> bytes:
    """Esporta le opportunità in formato Excel"""
    
    if not filename:
        filename = f"reddit_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Prepara dati per export
    export_data = []
    for opp in opportunities:
        strategy = opp.get('engagement_strategy', {})
        export_data.append({
            'Overall Score': opp['overall_score'],
            'Business Alignment': opp['business_alignment_score'],
            'Sensitivity': opp['sensitivity_score'],
            'Priority': opp['priority_score'],
            'Subreddit': opp['subreddit'],
            'Title': opp['title'],
            'Content': opp['selftext'][:500] if opp['selftext'] else '',
            'Author': opp['author'],
            'Reddit URL': opp['reddit_url'],
            'Comments': opp['num_comments'],
            'Approach': strategy.get('approach', ''),
            'Recommended Tone': strategy.get('tone', ''),
            'Key Points': '|'.join(strategy.get('key_points', [])),
            'Avoid': '|'.join(strategy.get('avoid', [])),
            'Suggested CTA': strategy.get('cta', ''),
            'Value Offered': strategy.get('value_proposition', ''),
            'Competitive Advantage': strategy.get('competitive_advantage', ''),
            'Problem Stage': strategy.get('problem_stage', ''),
            'Estimated Impact': strategy.get('impact', ''),
            'Created': opp['created_utc'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(opp['created_utc'], datetime) else str(opp['created_utc'])
        })
    
    df = pd.DataFrame(export_data)
    
    # Crea file Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Opportunities', index=False)
        
        # Formatta le colonne
        worksheet = writer.sheets['Opportunities']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    return output.getvalue()


def main():
    st.title("🌍 Reddit AI Business Scraper - Multi-Language Edition")
    st.markdown("### Advanced strategic opportunity identification system")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 API Configuration")
        
        reddit_client_id = st.text_input("Reddit Client ID", type="password", key="reddit_id")
        reddit_client_secret = st.text_input("Reddit Client Secret", type="password", key="reddit_secret")
        reddit_user_agent = st.text_input("Reddit User Agent", value="BusinessScraper/3.0", key="reddit_agent")
        openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        
        apis_configured = all([reddit_client_id, reddit_client_secret, reddit_user_agent, openai_api_key])
        
        if apis_configured:
            st.success("✅ Credentials configured")
        else:
            st.warning("⚠️ Enter all credentials")
        
        st.header("🌐 Language Settings")
        selected_language = st.selectbox(
            "Select Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: SUPPORTED_LANGUAGES[x]['name'],
            key="language"
        )
        
        # Checkpoint management
        st.header("💾 Checkpoint Management")
        checkpoint_manager = CheckpointManager()
        checkpoints = checkpoint_manager.list_checkpoints()
        
        if checkpoints:
            selected_checkpoint = st.selectbox("Load Checkpoint", ["None"] + checkpoints)
            if selected_checkpoint != "None" and st.button("Load"):
                data = checkpoint_manager.load_checkpoint(selected_checkpoint)
                if data:
                    st.success("✅ Checkpoint loaded")
                    st.session_state.checkpoint_data = data
    
    # Initialize scraper with selected language
    scraper = RedditAIBusinessScraper(language=selected_language)
    
    if not apis_configured:
        st.info("👈 Enter API credentials in sidebar to start")
        return
    
    # Configure APIs
    if not scraper.initialize_apis(reddit_client_id, reddit_client_secret, reddit_user_agent, openai_api_key):
        st.error("❌ API configuration error")
        return
    
    # Business configuration
    st.header("🏢 Business Context Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand_name = st.text_input("🏷️ Brand/Company Name", key="brand")
        about_us = st.text_area("📝 About Us", height=150, key="about")
    
    with col2:
        services = st.text_area("📋 Services Offered", height=150, key="services")
    
    if brand_name and about_us and services:
        scraper.business_info = {
            'brand_name': brand_name,
            'about_us': about_us,
            'services': services,
            'value_proposition': ''
        }
    
    # Search configuration
    st.header("🎯 Search Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        keywords_input = st.text_input(
            "🔍 Keywords (comma-separated)",
            placeholder="mortgage, personal loan, investment",
            key="keywords"
        )
        
        base_topics = [t.strip() for t in keywords_input.split(',') if t.strip()]
    
    with col2:
        time_filter = st.selectbox(
            "⏰ Time Period",
            options=['week', 'month', 'year'],
            index=1,
            key="time_filter"
        )
        
        # Subreddit selection
        use_suggested = st.checkbox(
            f"Use suggested subreddits for {SUPPORTED_LANGUAGES[selected_language]['name']}",
            value=True
        )
        
        if use_suggested:
            target_subreddits = SUPPORTED_LANGUAGES[selected_language]['subreddits']
        else:
            custom_subreddits = st.text_input("Custom subreddits (comma-separated)")
            target_subreddits = [s.strip() for s in custom_subreddits.split(',') if s.strip()] or ['all']
    
    with col3:
        min_overall_score = st.slider(
            "🎯 Minimum Score Required",
            min_value=50,
            max_value=80,
            value=60,
            step=5,
            key="min_score"
        )
    
    # Start search
    if st.button("🚀 Start Multi-Agent Analysis", type="primary") and base_topics:
        
        if not (brand_name and about_us and services):
            st.error("❌ Complete business configuration first!")
            return
        
        # Update session logger
        scraper.session_logger.update_session(
            language=selected_language,
            keywords=base_topics,
            filters_applied={
                'min_score': min_overall_score,
                'time_filter': time_filter,
                'subreddits': target_subreddits
            }
        )
        
        st.header("📊 Analysis Results")
        
        # Phase 1: Reddit search
        st.subheader(f"🌐 Phase 1: Searching Reddit ({SUPPORTED_LANGUAGES[selected_language]['name']})")
        with st.spinner("Searching posts..."):
            all_posts = scraper.search_reddit_with_checkpoint(
                base_topics, 
                time_filter, 
                limit_per_term=30, 
                subreddits=target_subreddits
            )
        
        if not all_posts:
            st.warning("❌ No posts found!")
            scraper.session_logger.save_session()
            return
        
        st.success(f"✅ Found {len(all_posts)} relevant posts")
        scraper.session_logger.update_session(posts_found=len(all_posts))
        
        # Phase 2: Multi-Agent analysis
        st.subheader("🤖 Phase 2: Multi-Agent Analysis")
        with st.spinner(f"Strategic analysis of {len(all_posts)} posts..."):
            strategic_opportunities, rejected_posts = scraper.evaluate_with_multi_agent_checkpoint(all_posts)
        
        # Apply additional filters
        filtered_opportunities = [
            opp for opp in strategic_opportunities 
            if opp['overall_score'] >= min_overall_score
        ]
        
        # Update session logger
        scraper.session_logger.update_session(
            opportunities_identified=len(filtered_opportunities)
        )
        scraper.session_logger.save_session()
        
        # Save results to session state
        st.session_state.analysis_results = {
            'opportunities': filtered_opportunities,
            'base_topics': base_topics,
            'brand_name': brand_name,
            'language': selected_language,
            'timestamp': datetime.now()
        }
        
        # Results display
        st.header("🎯 Final Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Posts Analyzed", len(all_posts))
        
        with col2:
            st.metric("✅ Strategic Opportunities", len(filtered_opportunities))
        
        with col3:
            if filtered_opportunities:
                avg_score = sum(o['overall_score'] for o in filtered_opportunities) / len(filtered_opportunities)
                st.metric("📈 Average Score", f"{avg_score:.1f}%")
            else:
                st.metric("📈 Average Score", "N/A")
        
        with col4:
            high_priority = len([o for o in filtered_opportunities if o['priority_score'] >= 7])
            st.metric("🔥 High Priority", high_priority)
    
    # Display results from session state (persists across downloads)
    if st.session_state.analysis_results:
        filtered_opportunities = st.session_state.analysis_results['opportunities']
        
        if filtered_opportunities:
            st.subheader(f"🏆 {len(filtered_opportunities)} Strategic Opportunities")
            
            # Display opportunities
            for i, opp in enumerate(filtered_opportunities[:20], 1):
                with st.expander(f"#{i}. [Score: {opp['overall_score']:.1f}%] {opp['title'][:80]}..."):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**📍 Subreddit:** r/{opp['subreddit']}")
                        st.write(f"**👤 Author:** {opp['author']}")
                        st.write(f"**💬 Comments:** {opp['num_comments']}")
                        
                        if opp['selftext']:
                            st.write("**📝 Content:**")
                            st.write(opp['selftext'][:300] + ("..." if len(opp['selftext']) > 300 else ""))
                    
                    with col2:
                        strategy = opp.get('engagement_strategy', {})
                        st.info(f"**Approach:** {strategy.get('approach', 'N/A')}")
                        st.info(f"**Impact:** {strategy.get('impact', 'N/A')}")
                    
                    st.markdown(f"[🔗 Go to Reddit post]({opp['reddit_url']})")
            
            # Export section
            st.subheader("💾 Export Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            timestamp = st.session_state.analysis_results['timestamp'].strftime("%Y%m%d_%H%M%S")
            
            with col1:
                # CSV export
                df = pd.DataFrame([{
                    'Score': opp['overall_score'],
                    'Subreddit': opp['subreddit'],
                    'Title': opp['title'],
                    'URL': opp['reddit_url']
                } for opp in filtered_opportunities])
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"reddit_opportunities_{timestamp}.csv",
                    mime="text/csv",
                    key="csv_download"
                )
            
            with col2:
                # Excel export
                excel_data = export_to_excel(filtered_opportunities)
                st.download_button(
                    label="📑 Download Excel",
                    data=excel_data,
                    file_name=f"reddit_opportunities_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="excel_download"
                )
            
            with col3:
                # JSON export
                json_data = {
                    'timestamp': timestamp,
                    'language': st.session_state.analysis_results['language'],
                    'keywords': st.session_state.analysis_results['base_topics'],
                    'opportunities': [{
                        **opp,
                        'created_utc': opp['created_utc'].isoformat() if isinstance(opp['created_utc'], datetime) else str(opp['created_utc'])
                    } for opp in filtered_opportunities]
                }
                
                json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"reddit_opportunities_{timestamp}.json",
                    mime="application/json",
                    key="json_download"
                )
            
            with col4:
                # Log export
                log_data = scraper.session_logger.export_to_excel()
                if log_data:
                    st.download_button(
                        label="📜 Download Session Log",
                        data=log_data,
                        file_name=f"session_log_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="log_download"
                    )

if __name__ == "__main__":
    main()
