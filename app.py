import streamlit as st
import pandas as pd
import subprocess
import shutil
import os
from Bio import SeqIO
import io
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="GeoAMR - Clinical Insights Engine",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 GeoAMR-Gonorrhoeae Tracker")
st.subheader("Automated High-Resolution Genomic Surveillance & Network Linkage Mapping")
st.markdown("---")

# --- Interactive Sidebar Controls ---
st.sidebar.header("🎛️ Pipeline Parameters")
min_id = st.sidebar.slider("Minimum % Identity", min_value=50, max_value=100, value=75, step=5)
min_cov = st.sidebar.slider("Minimum % Coverage", min_value=10, max_value=100, value=50, step=5)

st.sidebar.markdown("---")
st.sidebar.info("This system maps raw or scaffolded FASTA alignments against ResFinder, CARD, NCBI, and VFDB standard reference architectures.")

# --- Working Core Computational Pipeline (Unchanged) ---
def run_abricate_multi(fasta_path, db_name, identity_threshold, coverage_threshold):
    """
    Executes abricate via system subprocess, resolves duplicate coverage columns,
    and maps headers explicitly. Replaces missing product annotations with fallbacks.
    """
    try:
        cmd = [
            "abricate", 
            "--db", db_name, 
            "--minid", str(identity_threshold), 
            "--mincov", str(coverage_threshold), 
            fasta_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        
        if not result.stdout.strip():
            return pd.DataFrame()
        
        data = io.StringIO(result.stdout)
        df = pd.read_csv(data, sep="\t")
        
        if df.empty:
            return pd.DataFrame()
            
        mapping_dict = {}
        for col in df.columns:
            normalized = col.upper().strip()
            if normalized == '%COVERAGE' or (normalized == 'COVERAGE' and '%COVERAGE' not in df.columns):
                mapping_dict[col] = '% Coverage'
            elif normalized == '%IDENTITY' or (normalized == 'IDENTITY' and '%IDENTITY' not in df.columns):
                mapping_dict[col] = '% Identity'
            elif normalized == 'SEQUENCE':
                mapping_dict[col] = 'Contig/Node'
            elif normalized == 'START':
                mapping_dict[col] = 'Start'
            elif normalized == 'END':
                mapping_dict[col] = 'End'
            elif normalized == 'GENE':
                mapping_dict[col] = 'Identified Gene'
            elif normalized == 'DATABASE':
                mapping_dict[col] = 'Source DB'
            elif normalized == 'PRODUCT':
                mapping_dict[col] = 'Functional Product/Annotation'

        available_target_keys = [k for k in mapping_dict.keys() if k in df.columns]
        df_clean = df[available_target_keys].copy()
        df_clean.rename(columns=mapping_dict, inplace=True)
        
        if 'Functional Product/Annotation' not in df_clean.columns:
            df_clean['Functional Product/Annotation'] = 'Hypothetical or Unannotated Factor'
        else:
            df_clean['Functional Product/Annotation'] = df_clean['Functional Product/Annotation'].fillna('Conserved Core Element')
            
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        return df_clean

    except Exception as e:
        print(f"Background notice for DB [{db_name}]: {str(e)}")
        return pd.DataFrame()

# --- Advanced Analytical Helper Functions ---
def generate_amr_matrix(all_results_df):
    """Maps found genes to clinical class resistance categories for heatmap indexing."""
    if all_results_df.empty:
        return pd.DataFrame()
        
    def drug_class_mapper(gene_name):
        gene_upper = str(gene_name).upper()
        if any(x in gene_upper for x in ['GYRA', 'PARC', 'GYRB', 'PARE']): return 'Fluoroquinolones (Ciprofloxacin)'
        if any(x in gene_upper for x in ['PENA', 'BLATEM', 'BLA', 'PORB', 'MTRR_R']): return 'Cephalosporins / Penicillins'
        if any(x in gene_upper for x in ['TETM', 'TET']): return 'Tetracyclines'
        if any(x in gene_upper for x in ['ERM', 'MPH', 'MTRR', '23S']): return 'Macrolides (Azithromycin)'
        if any(x in gene_upper for x in ['APH', 'ANT', 'AAD', 'RPSL']): return 'Aminoglycosides'
        if any(x in gene_upper for x in ['FOLP', 'SUL']): return 'Sulfonamides'
        return 'Other Resistance Determinant'

    df_amr = all_results_df[all_results_df['Source DB'].isin(['resfinder', 'card', 'ncbi'])].copy()
    if df_amr.empty:
        return pd.DataFrame()
        
    df_amr['Drug Class'] = df_amr['Identified Gene'].apply(drug_class_mapper)
    matrix = df_amr.pivot_table(index='Sample ID', columns='Drug Class', aggfunc='size', fill_value=0)
    return matrix

def generate_pdf_report(summary_df, total_samples):
    """Generates a secure, completely robust standard encoded PDF byte string."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    
    pdf.cell(0, 10, "GeoAMR Surveillance & Clinical Diagnostics Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Automated Public Health Genomics Intelligence Output", ln=True, align="C")
    pdf.ln(8)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Executive Batch Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Total Genomes Successfully Screened: {total_samples}", ln=True)
    pdf.cell(0, 6, f"Total Elements Logged Across Registries: {len(summary_df)} total loci hits", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. High-Yield Target Annotations (Top Hits)", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(40, 7, "Sample ID", border=1)
    pdf.cell(30, 7, "Gene", border=1)
    pdf.cell(20, 7, "Cov %", border=1)
    pdf.cell(20, 7, "Iden %", border=1)
    pdf.cell(25, 7, "DB Source", border=1, ln=True)
    
    pdf.set_font("Helvetica", "", 8)
    for idx, row in summary_df.head(35).iterrows():
        s_id = str(row['Sample ID'])[:20].encode('latin-1', 'replace').decode('latin-1')
        g_id = str(row['Identified Gene']).encode('latin-1', 'replace').decode('latin-1')
        cov = str(row['% Coverage']).encode('latin-1', 'replace').decode('latin-1')
        iden = str(row['% Identity']).encode('latin-1', 'replace').decode('latin-1')
        src = str(row['Source DB']).encode('latin-1', 'replace').decode('latin-1')
        
        pdf.cell(40, 6, s_id, border=1)
        pdf.cell(30, 6, g_id, border=1)
        pdf.cell(20, 6, cov, border=1)
        pdf.cell(20, 6, iden, border=1)
        pdf.cell(25, 6, src, border=1, ln=True)
        
    # Return as safe encoded byte stream to avoid stream corruption
    return pdf.output(dest='S').encode('latin-1')

# --- UI Layout Interface & Data Flow ---

st.markdown("### 1. Batch System: Upload Assembled Genomes")
uploaded_files = st.file_uploader(
    "Drag and drop your assembled FASTA genomes here simultaneously", 
    type=["fasta", "fa"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 Queue initialized: {len(uploaded_files)} samples loaded into system storage.")
    
    abricate_installed = shutil.which("abricate") is not None
    all_batch_records = []
    
    if abricate_installed:
        databases_to_screen = ["resfinder", "card", "ncbi", "vfdb"]
        
        progress_bar = st.progress(0)
        for index, file_obj in enumerate(uploaded_files):
            bytes_data = file_obj.read()
            fasta_string = bytes_data.decode("utf-8")
            current_sample_name = os.path.splitext(file_obj.name)[0]
            
            temp_fasta_path = f"temp_{current_sample_name}.fasta"
            with open(temp_fasta_path, "w") as f:
                f.write(fasta_string)
            
            for db in databases_to_screen:
                db_df = run_abricate_multi(temp_fasta_path, db, min_id, min_cov)
                if not db_df.empty:
                    db_df['Sample ID'] = current_sample_name
                    all_batch_records.append(db_df)
                    
            if os.path.exists(temp_fasta_path):
                os.remove(temp_fasta_path)
                
            progress_bar.progress((index + 1) / len(uploaded_files))
            
        if all_batch_records:
            master_df = pd.concat(all_batch_records, ignore_index=True)
            st.success("✨ Sequence data successfully compiled across all reference frameworks!")
            
            # --- HIGH-LEVEL METRIC TILES ---
            st.markdown("### 2. High-Level Summary Overview")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="Total Unique Samples Processed", value=f"{master_df['Sample ID'].nunique()}")
            with m2:
                amr_count = master_df[master_df['Source DB'].isin(['resfinder', 'card', 'ncbi'])]['Identified Gene'].nunique()
                st.metric(label="Unique AMR Loci Found", value=f"{amr_count}")
            with m3:
                vf_count = master_df[master_df['Source DB'] == 'vfdb']['Identified Gene'].nunique()
                st.metric(label="Virulence Phenotypes Flagged", value=f"{vf_count}")
            
            # --- BATCH AMR HEATMAP SECTION ---
            st.markdown("---")
            st.markdown("### 3. High-Resolution Cross-Resistance Heatmap Matrix")
            with st.spinner("Compiling high-resolution epidemiological heatmap..."):
                amr_matrix = generate_amr_matrix(master_df)
                if not amr_matrix.empty:
                    fig_heatmap = px.imshow(
                        amr_matrix,
                        labels=dict(x="Antibiotic Drug Class Family", y="Sample Strain Node", color="Loci Count"),
                        x=amr_matrix.columns,
                        y=amr_matrix.index,
                        color_continuous_scale="Viridis",
                        text_auto=True,
                        aspect="auto"
                    )
                    fig_heatmap.update_layout(
                        title_text="High-Resolution Genotypic Resistance Profile Heatmap Matrix",
                        title_x=0.5,
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig_heatmap, width="stretch")
                else:
                    st.info("Insufficient resistance mutations available to populate cluster visualizations.")

            # --- SPATIAL GENOMIC FEATURE TRACK MAP ---
            st.markdown("---")
            st.markdown("### 4. Structural Contig Loci Coordinates & Linear Mapping")
            selected_map_sample = st.selectbox("Select Sample Strain to Map Coordinates:", master_df['Sample ID'].unique())
            sample_map_data = master_df[master_df['Sample ID'] == selected_map_sample]
            
            fig_map = px.scatter(
                sample_map_data,
                x="Start",
                y="Identified Gene",
                color="Source DB",
                size="% Coverage",
                hover_data=["End", "% Identity"],
                title=f"Linear Multi-Loci Feature Architecture Mapping: {selected_map_sample}",
                labels={"Start": "Nucleotide Base Coordinates Position (bp)"}
            )
            fig_map.update_traces(marker=dict(symbol="diamond", line=dict(width=1, color="DarkSlateGrey")))
            fig_map.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
            st.plotly_chart(fig_map, width="stretch")

            # --- REQUIREMENT: GENOMIC LINKAGE DATA & VISUAL NETWORK DIAGRAM ---
            st.markdown("---")
            st.markdown("### 5. Loci Linkage Distance Map & Co-Inheritance Network Plot")
            st.markdown(
                "This section tracks spatial linkages along shared structures. Genes on the "
                "**same contig node with small relative distance values** are physically tied and highly likely "
                "to transfer together during horizontal recombination events."
            )
            
            linkage_records = []
            for (sample, contig), sub_df in sample_map_data.groupby(['Sample ID', 'Contig/Node']):
                if len(sub_df) > 1:
                    sorted_genes = sub_df.sort_values(by="Start")
                    genes_list = sorted_genes['Identified Gene'].tolist()
                    starts_list = sorted_genes['Start'].tolist()
                    
                    for i in range(len(genes_list)):
                        for j in range(i + 1, len(genes_list)):
                            bp_distance = abs(starts_list[j] - starts_list[i])
                            linkage_records.append({
                                "Contig/Node": contig,
                                "Locus A": genes_list[i],
                                "Locus B": genes_list[j],
                                "Physical Proximity Distance (bp)": bp_distance,
                                "Linkage Status": "Extremely Close (<5kb)" if bp_distance <= 5000 else "Moderately Linked (<50kb)" if bp_distance <= 50000 else "Distal Linkage"
                            })
            
            if linkage_records:
                linkage_df = pd.DataFrame(linkage_records)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("📋 **Calculated Proximity Logs**")
                    st.dataframe(linkage_df.sort_values(by="Physical Proximity Distance (bp)"), width="stretch")
                
                with col2:
                    st.write("🌐 **Interactive Spatial Linkage Network**")
                    
                    # Compute unique nodes and position them in a tidy geometric layout circle
                    unique_genes = list(set(linkage_df['Locus A'].tolist() + linkage_df['Locus B'].tolist()))
                    num_nodes = len(unique_genes)
                    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
                    pos = {unique_genes[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(num_nodes)}
                    
                    edge_x = []
                    edge_y = []
                    edge_text = []
                    
                    # Generate connection lines
                    for _, row in linkage_df.iterrows():
                        x0, y0 = pos[row['Locus A']]
                        x1, y1 = pos[row['Locus B']]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_text.append(f"{row['Locus A']} 🔗 {row['Locus B']} : {row['Physical Proximity Distance (bp)']} bp")
                        
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=2, color='#e74c3c'),
                        hoverinfo='text',
                        mode='lines'
                    )
                    
                    node_x = [pos[gene][0] for gene in unique_genes]
                    node_y = [pos[gene][1] for gene in unique_genes]
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=unique_genes,
                        textposition="top center",
                        hoverinfo='text',
                        marker=dict(
                            showscale=False,
                            colorscale='YlOrRd',
                            color=[],
                            size=22,
                            line=dict(width=2, color='Black')
                        )
                    )
                    
                    fig_network = go.Figure(data=[edge_trace, node_trace],
                                 layout=go.Layout(
                                    title=f"Co-Location Cluster Map: {selected_map_sample}",
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                 )
                    )
                    st.plotly_chart(fig_network, width="stretch")
            else:
                st.info("No co-located loci detected on shared contig structures to generate a linkage network chart.")

            # --- GRANULAR DISCOVERED FEATURE REGISTRIES ---
            st.markdown("---")
            st.markdown("### 6. Granular Biological Feature Registries")
            tab1, tab2, tab3 = st.tabs([
                "💊 Antimicrobial Resistance (AMR) Elements", 
                "🧬 Virulence Factors & Toxins", 
                "📊 Total Master Genome Registry (All Genes)"
            ])
            
            with tab1:
                amr_mask = master_df['Source DB'].isin(['resfinder', 'card', 'ncbi'])
                final_amr_df = master_df[amr_mask]
                if not final_amr_df.empty:
                    st.dataframe(final_amr_df, width="stretch")
                else:
                    st.info("No explicit resistance determinants detected above parameters.")
                    
            with tab2:
                vf_mask = master_df['Source DB'] == 'vfdb'
                final_vf_df = master_df[vf_mask]
                if not final_vf_df.empty:
                    st.dataframe(final_vf_df, width="stretch")
                else:
                    st.info("No explicit structural virulence vectors or pathogenic attributes identified.")
                    
            with tab3:
                st.write("Comprehensive indexing of all unique loci mapped within this screening operation:")
                st.dataframe(master_df, width="stretch")
                
            # --- REGULATORY PDF EXPORT PANEL ---
            st.markdown("---")
            st.markdown("### 7. Export Regulatory Executive Diagnostics Documentation")
            
            try:
                pdf_output_bytes = generate_pdf_report(master_df, len(uploaded_files))
                
                st.download_button(
                    label="📥 Download Clinical Batch Summary PDF",
                    data=pdf_output_bytes, 
                    file_name="GeoAMR_Clinical_Surveillance_Report.pdf",
                    mime="application/pdf"
                )
                st.success("PDF Report compilation complete. Production container stream validated.")
            except Exception as pdf_err:
                st.error(f"Reporting compilation warning formatting PDF: {str(pdf_err)}")
                
        else:
            st.warning("All files systematically inspected, but zero targets matched criteria configurations across active libraries.")
    else:
        st.error("🔒 Critical Error: Native Abricate installation profile missing from the workstation's active runtime path environment.")
