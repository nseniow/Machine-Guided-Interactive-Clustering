// This function takes the values from the InputForm and transforms them into cookies.
// These cookies will be used to pre-populate the fields in InputForm next time it is used.
export function updateCookies(values) {
    document.cookie = "questionsPerIteration=" + values.questionsPerIteration;
    document.cookie = "numberOfClusters=" + values.numberOfClusters;
    document.cookie = "maxConstraintPercent=" + values.maxConstraintPercent;
    document.cookie = "reduction_algorithm=" + values.reduction_algorithm;
    document.cookie = "checked=" + values.checked;
}