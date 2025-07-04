import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="CorestratAI Brand Watcher",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .trending-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .growth-positive {
        color: #28a745;
        font-weight: bold;
    }
    .growth-negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_hot_brands(days: int = 7, brands: List[str] = None) -> Dict:
    """Fetch hot brands from API"""
    try:
        payload = {"days": days}
        if brands:
            payload["brands"] = brands
        
        response = requests.post(f"{API_BASE_URL}/hot-brands", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_available_brands() -> List[str]:
    """Fetch available brands from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/brands")
        if response.status_code == 200:
            return response.json()["brands"]
        return []
    except:
        return []

def format_growth_rate(growth_rate: float) -> str:
    """Format growth rate with color coding"""
    if growth_rate > 0:
        return f'<span class="growth-positive">+{growth_rate:.1f}%</span>'
    elif growth_rate < 0:
        return f'<span class="growth-negative">{growth_rate:.1f}%</span>'
    else:
        return "0.0%"

def create_score_chart(brands_data: List[Dict]) -> go.Figure:
    """Create a horizontal bar chart for brand scores"""
    df = pd.DataFrame(brands_data)
    
    fig = go.Figure()
    
    # Add trending scores
    fig.add_trace(go.Bar(
        y=df['brand'],
        x=df['trending_score'],
        name='Trending Score',
        orientation='h',
        marker_color='#ff6b6b',
        text=df['trending_score'].round(1),
        textposition='inside'
    ))
    
    # Add absolute scores
    fig.add_trace(go.Bar(
        y=df['brand'],
        x=df['absolute_score'],
        name='Absolute Score',
        orientation='h',
        marker_color='#1f77b4',
        text=df['absolute_score'].round(1),
        textposition='inside',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Brand Scores Comparison",
        xaxis_title="Score",
        yaxis_title="Brands",
        barmode='overlay',
        height=max(400, len(brands_data) * 30),
        showlegend=True
    )
    
    return fig

def create_metrics_breakdown(brand_data: Dict) -> go.Figure:
    """Create a radar chart for metrics breakdown"""
    categories = ['News Count', 'Twitter Buzz', 'PR Coverage', 'Trends Score']
    values = [
        brand_data['news_count'],
        brand_data['twitter_count'],
        brand_data['pr_count'],
        brand_data['trends_score']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=brand_data['brand'],
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1] if max(values) > 0 else [0, 1]
            )),
        showlegend=True,
        title=f"Metrics Breakdown: {brand_data['brand']}"
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 CorestratAI Brand Watcher</h1>', unsafe_allow_html=True)
    st.markdown("### Track trending retail brands in India with AI-powered insights")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Time period selection
    days = st.sidebar.selectbox(
        "Analysis Period",
        options=[7, 14, 30],
        index=0,
        help="Number of days to analyze"
    )
    
    # Brand selection
    available_brands = fetch_available_brands()
    if available_brands:
        selected_brands = st.sidebar.multiselect(
            "Select Brands (leave empty for all)",
            options=available_brands,
            help="Choose specific brands to analyze"
        )
    else:
        selected_brands = []
        st.sidebar.warning("Could not load brands list")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    with st.spinner("Fetching brand data..."):
        data = fetch_hot_brands(days, selected_brands if selected_brands else None)
    
    if not data:
        st.error("Failed to fetch data. Please check if the API server is running.")
        st.info("To start the API server, run: `uvicorn api.main:app --reload`")
        return
    
    brands_data = data["brands"]
    
    if not brands_data:
        st.warning("No brand data available for the selected period.")
        return
    
    # Summary metrics
    st.header("📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Brands Analyzed",
            len(brands_data),
            help="Number of brands in the analysis"
        )
    
    with col2:
        avg_trending_score = sum(b['trending_score'] for b in brands_data) / len(brands_data)
        st.metric(
            "Avg Trending Score",
            f"{avg_trending_score:.1f}",
            help="Average trending score across all brands"
        )
    
    with col3:
        total_mentions = sum(b['news_count'] + b['twitter_count'] + b['pr_count'] for b in brands_data)
        st.metric(
            "Total Mentions",
            f"{total_mentions:,}",
            help="Total mentions across all channels"
        )
    
    with col4:
        growing_brands = len([b for b in brands_data if b['growth_rate'] > 0])
        st.metric(
            "Growing Brands",
            f"{growing_brands}/{len(brands_data)}",
            help="Brands with positive growth rate"
        )
    
    # Top brands section
    st.header("🔥 Hot Brands This Week")
    
    # Display top 10 brands
    top_brands = brands_data[:10]
    
    for i, brand in enumerate(top_brands, 1):
        with st.expander(f"#{i} {brand['brand']} - Score: {brand['trending_score']:.1f}", expanded=i <= 3):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**AI Summary:** {brand['summary']}")
                
                # Metrics
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    st.metric("News", brand['news_count'])
                with metrics_col2:
                    st.metric("Twitter", brand['twitter_count'])
                with metrics_col3:
                    st.metric("PR Coverage", brand['pr_count'])
                with metrics_col4:
                    st.metric("Trends", f"{brand['trends_score']:.1f}")
                
                # Growth rate
                growth_html = format_growth_rate(brand['growth_rate'])
                st.markdown(f"**Growth Rate:** {growth_html}", unsafe_allow_html=True)
            
            with col2:
                # Metrics breakdown chart
                fig = create_metrics_breakdown(brand)
                st.plotly_chart(fig, use_container_width=True)
    
    # Comparison chart
    st.header("📊 Brand Comparison")
    if len(brands_data) > 1:
        fig = create_score_chart(brands_data[:15])  # Top 15 brands
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.header("📋 Detailed Data")
    df = pd.DataFrame(brands_data)
    df = df[['brand', 'trending_score', 'absolute_score', 'news_count', 
             'twitter_count', 'pr_count', 'trends_score', 'growth_rate']]
    df.columns = ['Brand', 'Trending Score', 'Absolute Score', 'News', 
                  'Twitter', 'PR', 'Trends', 'Growth %']
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Trending Score": st.column_config.NumberColumn(format="%.1f"),
            "Absolute Score": st.column_config.NumberColumn(format="%.1f"),
            "Trends": st.column_config.NumberColumn(format="%.1f"),
            "Growth %": st.column_config.NumberColumn(format="%.1f%%"),
        }
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**Last Updated:** {data['generated_at']} | "
        f"**Analysis Period:** {data['period_days']} days | "
        f"**Powered by:** CorestratAI"
    )

if __name__ == "__main__":
    main()
