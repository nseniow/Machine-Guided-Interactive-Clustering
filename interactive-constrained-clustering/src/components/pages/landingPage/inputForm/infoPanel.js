import React, { Component } from 'react';

export default class InfoPanel extends Component {
    FormattedInfo = (title, body, url, warning) => {
        return (
            <div class="my-4 mx-4">
                <h5 class="font-weight-bold">{title}</h5>
                <p>
                    {body} 
                    {url ? <a href={url} target="_blank" class="text-secondary small"> Learn More</a> : ""}
                    {warning ? <p class="text-danger">Note: {warning}</p> : ""}
                </p>
            </div>
            
        )
    }

    render() {
        return (        
            <div class="panel panel-default bg-light my-4 rounded border border-info">
                <div class="panel-heading bg-info d-flex justify-content-between">
                    <h3 class="panel-title py-2 px-2">Definitions</h3>
                    <button onClick={() => this.props.showInfo()} type="button" class="btn btn-info btn-lg">X</button>
                </div>
                <div class="panel-body my-2 mx-2 overflow-auto" style={{maxHeight: "88vh"}}>
                    {this.FormattedInfo("Dataset", "Upload a CSV file with numerical data that you wish to cluster. The first row must contain feature/column headers, and the classifier column must be removed.")}
                    <hr/>
                    {this.FormattedInfo("Questions per Iteration", "The number of constraint questions (must-link / cannot-link) per iteration of clustering.")}
                    <hr/>
                    {this.FormattedInfo("Max Constraint Percentage", "Determines how many rounds of questions/clustering will take place.")}
                    <hr/>
                    {this.FormattedInfo("Number of Clusters", "The number of classes the data will be partitioned into.")}
                    <hr/>
                    {this.FormattedInfo("Dimensionality Reduction Algorithm", "A process that transforms high-dimensional data into a lower-dimensional space while preserving the essence of the original data. The following three dimensionality reduction algorithms are offered: TSNE, UMAP, and PCA.")}
                    {this.FormattedInfo("TSNE", "(T-distributed Stochastic Neighbor Embedding) is a reduction technique that finds the similarity measure between pairs of instances, well suited for the visualization of high-dimensional datasets.", "https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding", "Known to experience underflow issues with large datasets.")}
                    {this.FormattedInfo("UMAP", "(Uniform Manifold Approximation and Projection) is similar to TSNE but is scalable - it can be applied directly to sparse matrices.", "https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection", "Known to experience underflow issues with large datasets.")}
                    {this.FormattedInfo("PCA", "(Principal Component Analysis) statistical technique that transforms data to describes variation in fewer dimensions than the original data.", "https://en.wikipedia.org/wiki/Principal_component_analysis")}
                    <hr/>
                    {this.FormattedInfo("Evaluation Algorithms", "Algorithms developed to assess the effectiveness of clustering methods.")}
                    {this.FormattedInfo("iNNE", "(Isolation using Nearest Neighbor Ensemble) is an isolation-based anomaly detector.", "https://onlinelibrary.wiley.com/doi/abs/10.1111/coin.12156")}
                    {this.FormattedInfo("COPOD", "(Copula-Based Outlier Detection) identifies data points that deviate from general distribution.", "https://arxiv.org/abs/2009.09463")}
                    {this.FormattedInfo("Isolation Forest", "An algorithm for anomaly detection in linear time.", "https://en.wikipedia.org/wiki/Isolation_forest")}
                    {this.FormattedInfo("LOF", "(Local Outlier Factor) finds anomalies by measuring local deviation of a point with respect to its neighbors.", "https://en.wikipedia.org/wiki/Local_outlier_factor")}
                    {this.FormattedInfo("Silhouette", "Validates the consistency within data clusters.", "https://en.wikipedia.org/wiki/Silhouette_(clustering)")}
                </div>
            </div>
        )
    }
}